from __future__ import annotations
from typing import Any
from sqlalchemy import select
from sqlalchemy.orm import Session
from app.hypothesis.signal_quality import normalize_revision_signal, persisted_signal_payload
from app.models import Experience,HypothesisEvidence,HypothesisRevision,ReflectionRecord,RevisionSignal,StrategyHypothesis

class RevisionSignalService:
    def __init__(self,db:Session)->None: self.db=db
    @staticmethod
    def _reflection_payload(reflection:ReflectionRecord)->dict[str,Any]:
        return {"insight":reflection.insight,"applicable_context":reflection.applicable_context or {},"proposed_change":reflection.strategy_signal,"expected_effect":reflection.expected_effect,"canonical_signal":reflection.canonical_signal}
    @staticmethod
    def _revision_payload(revision:HypothesisRevision|None)->dict[str,Any]:
        if revision is None:return {}
        return {"revision":revision.revision_number,"insight":revision.insight,"when_data":revision.when_data or {},"do_data":revision.do_data or {},"expect_data":revision.expect_data or {},"applicable_context":revision.when_data or {},"proposed_change":(revision.do_data or {}).get("text",revision.do_data or {}),"expected_effect":(revision.expect_data or {}).get("text",revision.expect_data or {}),"canonical_signal":revision.canonical_signal}
    def _validation_context(self,*,hypothesis_id:int,base_revision:int,reflection_id:int)->tuple[dict[str,Any],dict[str,Any],bool]:
        reflection=self.db.get(ReflectionRecord,reflection_id)
        revision=self.db.scalar(select(HypothesisRevision).where(HypothesisRevision.hypothesis_id==hypothesis_id,HypothesisRevision.revision_number==base_revision))
        explicit_feedback=False
        if reflection is not None and reflection.experience_id is not None:
            experience=self.db.get(Experience,reflection.experience_id)
            explicit_feedback=bool(experience is not None and experience.user_feedback and str(experience.user_feedback).strip())
        return (self._reflection_payload(reflection) if reflection is not None else {},self._revision_payload(revision),explicit_feedback)
    def _is_valid_persisted(self,signal:RevisionSignal)->tuple[bool,str]:
        rp,hp,feedback=self._validation_context(hypothesis_id=signal.hypothesis_id,base_revision=signal.base_revision,reflection_id=signal.reflection_id)
        normalized,reason=normalize_revision_signal(persisted_signal_payload(signal),reflection_payload=rp,hypothesis_payload=hp,explicit_user_feedback=feedback)
        return normalized is not None,reason
    def list_all(self,*,hypothesis_id:int|None=None,base_revision:int|None=None,valid_only:bool=True)->list[RevisionSignal]:
        stmt=select(RevisionSignal)
        if hypothesis_id is not None: stmt=stmt.where(RevisionSignal.hypothesis_id==hypothesis_id)
        if base_revision is not None: stmt=stmt.where(RevisionSignal.base_revision==base_revision)
        stmt=stmt.order_by(RevisionSignal.created_at.asc(),RevisionSignal.id.asc())
        signals=list(self.db.scalars(stmt).all())
        return signals if not valid_only else [s for s in signals if self._is_valid_persisted(s)[0]]
    def list_invalid(self)->list[tuple[RevisionSignal,str]]:
        out=[]
        for s in self.list_all(valid_only=False):
            valid,reason=self._is_valid_persisted(s)
            if not valid: out.append((s,reason))
        return out
    def count_invalid(self)->int:return len(self.list_invalid())
    def create_from_refines(self,*,hypothesis:StrategyHypothesis,reflection:ReflectionRecord,evidence:HypothesisEvidence,raw_signals:list[dict[str,Any]])->list[RevisionSignal]:
        if evidence.relation_type!="REFINES": return []
        target_revision=int(evidence.hypothesis_revision)
        revision=self.db.scalar(select(HypothesisRevision).where(HypothesisRevision.hypothesis_id==hypothesis.id,HypothesisRevision.revision_number==target_revision))
        experience=None;interaction_id=None;explicit_feedback=False
        if reflection.experience_id is not None:
            experience=self.db.get(Experience,reflection.experience_id)
            if experience is not None:
                interaction_id=experience.interaction_id
                explicit_feedback=bool(experience.user_feedback and str(experience.user_feedback).strip())
        rp=self._reflection_payload(reflection);hp=self._revision_payload(revision);created=[]
        for raw in raw_signals:
            normalized,_=normalize_revision_signal(raw,reflection_payload=rp,hypothesis_payload=hp,explicit_user_feedback=explicit_feedback)
            if normalized is None: continue
            dimension=normalized["dimension"];operation=normalized["operation"];target_field=normalized["target_field"]
            existing=self.db.scalar(select(RevisionSignal).where(RevisionSignal.hypothesis_id==hypothesis.id,RevisionSignal.base_revision==target_revision,RevisionSignal.reflection_id==reflection.id,RevisionSignal.dimension==dimension,RevisionSignal.operation==operation,RevisionSignal.target_field==target_field))
            if existing is not None: continue
            signal=RevisionSignal(hypothesis_id=hypothesis.id,base_revision=target_revision,reflection_id=reflection.id,evidence_id=evidence.id,interaction_id=interaction_id,dimension=dimension,operation=operation,target_field=target_field,before_data=normalized["before_data"],observed_data=normalized["observed_data"],canonical_change=normalized["canonical_change"],confidence=normalized["confidence"])
            self.db.add(signal);created.append(signal)
        if created:
            self.db.commit()
            for s in created:self.db.refresh(s)
        return created
