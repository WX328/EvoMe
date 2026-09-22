from app.database import SessionLocal
from app.models import InteractionRecord, Experience, ReflectionRecord, HypothesisEvidence, RevisionSignal, StrategyHypothesis

db = SessionLocal()
try:
    i = db.get(InteractionRecord, 28)
    print("INTERACTION", i.id if i else None, i.user_input if i else None)
    e = db.query(Experience).filter(Experience.interaction_id == 28).order_by(Experience.id.desc()).first()
    print("EXPERIENCE", e.id if e else None, e.task_type if e else None, e.context_data if e else None)
    r = db.query(ReflectionRecord).filter(ReflectionRecord.experience_id == e.id).order_by(ReflectionRecord.id.desc()).first() if e else None
    print("REFLECTION", r.id if r else None, r.insight if r else None, r.strategy_signal if r else None, r.applicable_context if r else None)
    he = db.query(HypothesisEvidence).filter(HypothesisEvidence.reflection_id == r.id).order_by(HypothesisEvidence.id.desc()).first() if r else None
    print("EVIDENCE", he.id if he else None, he.relation_type if he else None, he.relation_confidence if he else None, he.relation_reason if he else None)
    sigs = db.query(RevisionSignal).filter(RevisionSignal.reflection_id == r.id).all() if r else []
    print("SIGNALS", [(s.id, s.dimension, s.operation, s.target_field, s.before_data, s.observed_data, s.canonical_change) for s in sigs])
    h = db.get(StrategyHypothesis, he.hypothesis_id) if he else None
    print("HYPOTHESIS", h.id if h else None, h.status if h else None, h.current_revision if h else None)
finally:
    db.close()
