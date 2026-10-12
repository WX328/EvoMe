import json

from app.database import Base, engine, SessionLocal
from app.migrations import migrate_v0_2, migrate_v0_2_1
from app.runtime import EvoMeRuntime
from app.cli_commands import COMMAND_NAMES, normalize_cli_input
from app.version import APP_VERSION


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    migrate_v0_2(engine)
    migrate_v0_2_1(engine)


def show_personal_memories(runtime: EvoMeRuntime) -> None:
    print("\n===== Personal Memory =====")
    memories = runtime.list_personal_memories()
    if not memories:
        print("暂无 Personal Memory")
    for m in memories:
        print(
            f"#{m.id} [{m.memory_type}] importance={m.importance:.2f} "
            f"confidence={m.confidence:.2f}\n{m.content}\n"
        )


def show_experiences(runtime: EvoMeRuntime) -> None:
    print("\n===== Evolution Experience =====")
    items = runtime.list_experiences()
    if not items:
        print("暂无 Experience")
    for e in items:
        context = e.context_data or {}
        print(
            f"#{e.id} [{e.task_type}] score={e.score:.2f} success={e.success}\n"
            f"Context: {context.get('task_type', 'unknown')} / "
            f"{context.get('complexity', 'unknown')} / "
            f"{context.get('interaction_mode', 'unknown')}\n"
            f"{e.result_summary}\n"
        )


def show_reflections(runtime: EvoMeRuntime) -> None:
    print("\n===== Reflection =====")
    items = runtime.list_reflections()
    if not items:
        print("暂无 Reflection")
    for r in items:
        insight = r.insight or r.lesson
        print(
            f"#{r.id} [{r.reflection_type}] confidence={r.confidence:.2f}\n"
            f"观察: {r.observation}\n"
            f"Insight: {insight}\n"
            f"Applicable Context: {r.applicable_context or {}}\n"
            f"Strategy Signal: {r.strategy_signal}\n"
            f"Skill Signal: {r.skill_signal}\n"
        )


def show_hypotheses(runtime: EvoMeRuntime) -> None:
    print("\n===== Strategy Hypotheses =====")
    items = runtime.list_hypotheses()
    if not items:
        print("暂无 Strategy Hypothesis")
    for h in items:
        pressure = runtime.get_revision_pressure(h.id)
        print(
            f"H#{h.id} [{h.status}] confidence={h.confidence:.2f} revision={h.current_revision}\n"
            f"Label: {h.title}\n"
            f"WHEN: {h.applicable_context or {}}\n"
            f"DO: {h.proposed_change}\n"
            f"EXPECT: {h.expected_effect}\n"
            f"Historical Evidence: support={h.supporting_count} refine={h.refinement_count} "
            f"counter={h.counter_count} independent={h.independent_evidence_count} "
            f"independent_support={h.independent_support_count} "
            f"diversity={h.context_diversity}\n"
            f"Revision Pressure: active_open={pressure.active_open} "
            f"ancestor_open={pressure.ancestor_open} resolved={pressure.resolved} "
            f"superseded={pressure.superseded} "
            f"inactive_branch={pressure.inactive_branch_open} invalid={pressure.invalid}\n"
        )


def show_revisions(runtime: EvoMeRuntime) -> None:
    print("\n===== Hypothesis Revisions =====")
    hypotheses = runtime.list_hypotheses()
    if not hypotheses:
        print("暂无 Hypothesis Revision")
        return
    for hypothesis in hypotheses:
        revisions = runtime.list_hypothesis_revisions(hypothesis.id)
        for revision in revisions:
            marker = " *current" if revision.revision_number == hypothesis.current_revision else ""
            print(
                f"H#{hypothesis.id}:r{revision.revision_number}{marker} "
                f"parent={('r' + str(revision.parent_revision)) if revision.parent_revision is not None else '-'} "
                f"reason={revision.created_reason}\n"
                f"WHEN: {revision.when_data or {}}\n"
                f"DO: {revision.do_data or {}}\n"
                f"EXPECT: {revision.expect_data or {}}\n"
            )


def show_revision_signals(runtime: EvoMeRuntime) -> None:
    print("\n===== Revision Signals =====")
    signals = runtime.list_revision_signals()
    invalid_count = runtime.count_invalid_revision_signals()
    if invalid_count:
        print(f"(已忽略 {invalid_count} 条旧版/无效 Revision Signal)")
    if not signals:
        print("暂无有效 Revision Signal")
        return
    for signal in signals:
        print(
            f"RS#{signal.id} -> H#{signal.hypothesis_id}:r{signal.base_revision} "
            f"reflection=R#{signal.reflection_id} "
            f"confidence={signal.confidence:.2f} "
            f"stored_state={signal.resolution_status or 'open'} "
            f"resolved_by=r{signal.resolved_by_revision if signal.resolved_by_revision is not None else '-'}\n"
            f"{signal.dimension} / {signal.operation} / {signal.target_field}\n"
            f"before: {signal.before_data}\n"
            f"observed: {signal.observed_data}\n"
            f"change: {signal.canonical_change}\n"
            f"provenance: {signal.provenance_data or {}}\n"
        )



def show_signal_revalidations(runtime: EvoMeRuntime) -> None:
    print("\n===== Ancestor Signal Revalidations =====")
    items = runtime.list_signal_revalidations()
    if not items:
        print("暂无 Ancestor Signal Revalidation")
        return
    for item in items:
        rebased = f"RS#{item.rebased_signal_id}" if item.rebased_signal_id is not None else "-"
        print(
            f"SRV#{item.id} H#{item.hypothesis_id}: r{item.source_revision} -> r{item.target_revision} "
            f"source=RS#{item.source_signal_id} outcome={item.outcome} "
            f"confidence={item.confidence:.2f} rebased={rebased}\n"
            f"reason: {item.reason}\n"
        )


def show_revision_groups(runtime: EvoMeRuntime) -> None:
    print("\n===== Revision Signal Groups =====")
    groups = runtime.list_revision_signal_groups()
    if not groups:
        print("暂无可聚合的 Revision Signal Group")
        return
    for group in groups:
        marker = " *analysis_candidate" if group.eligible_for_analysis else ""
        print(
            f"H#{group.hypothesis_id}:r{group.base_revision} "
            f"{group.dimension}/{group.operation}/{group.target_field}{marker}\n"
            f"signals={group.signal_count} independent={group.independent_count} "
            f"avg_confidence={group.average_confidence:.2f}\n"
            f"signal_ids={group.signal_ids} observed={group.observed_values}\n"
        )


def show_evidence_summary(runtime: EvoMeRuntime) -> None:
    print("\n===== Evidence Summary =====")
    hypotheses = runtime.list_hypotheses()
    if not hypotheses:
        print("暂无 Strategy Hypothesis")
        return
    for hypothesis in hypotheses:
        summary = runtime.get_evidence_summary(hypothesis.id)
        counts = summary["evidence_counts"]
        print(
            f"H#{hypothesis.id}:r{summary['revision']} confidence={summary['confidence']:.2f}\n"
            f"Evidence: support={counts['supports']} refine={counts['refines']} "
            f"counter={counts['conflicts']} independent={counts['independent_evidence']}\n"
            f"Context Coverage: {summary['context_coverage']}\n"
            f"Revision Pressure: active_open={summary['revision_pressure']['active_open']} "
            f"ancestor_open={summary['revision_pressure']['ancestor_open']} "
            f"resolved={summary['revision_pressure']['resolved']} "
            f"superseded={summary['revision_pressure']['superseded']}\n"
            f"Revision Groups: {len(summary['revision_groups'])} "
            f"analysis_candidates={len(summary['analysis_candidates'])}\n"
        )


def show_revision_pressure(runtime: EvoMeRuntime) -> None:
    print("\n===== Revision Pressure =====")
    hypotheses = runtime.list_hypotheses()
    if not hypotheses:
        print("暂无 Strategy Hypothesis")
        return
    for hypothesis in hypotheses:
        p = runtime.get_revision_pressure(hypothesis.id)
        print(
            f"H#{hypothesis.id}:r{p.current_revision} status={hypothesis.status} "
            f"lineage={p.lineage}\n"
            f"active_open={p.active_open} ids={p.active_open_ids}\n"
            f"ancestor_open={p.ancestor_open} ids={p.ancestor_open_ids}\n"
            f"resolved={p.resolved} ids={p.resolved_ids}\n"
            f"superseded={p.superseded}\n"
            f"inactive_branch_open={p.inactive_branch_open} "
            f"ids={p.inactive_branch_open_ids} invalid={p.invalid}\n"
        )


def show_revision_analyses(runtime: EvoMeRuntime) -> None:
    print("\n===== Revision Analyses =====")
    items = runtime.list_revision_analyses()
    if not items:
        print("暂无 Revision Analysis")
        return
    for item in items:
        verdict = "consistent" if item.consistent else "heterogeneous"
        print(
            f"RA#{item.id} -> H#{item.hypothesis_id}:r{item.base_revision} "
            f"{item.dimension}/{item.operation}/{item.target_field} "
            f"[{verdict}] confidence={item.direction_confidence:.2f}\n"
            f"signals={item.signal_ids}\n"
            f"reason: {item.reason}\n"
        )


def show_revision_proposals(runtime: EvoMeRuntime) -> None:
    print("\n===== Revision Proposals =====")
    items = runtime.list_revision_proposals()
    if not items:
        print("暂无 Revision Proposal")
        return
    for item in items:
        print(
            f"RP#{item.id} [{item.status}] -> H#{item.hypothesis_id}:r{item.base_revision} "
            f"{item.dimension}/{item.operation}/{item.target_field} "
            f"confidence={item.direction_confidence:.2f} kind={item.proposal_kind or 'raw'} "
            f"parent={item.parent_proposal_id or '-'} norm=v{item.normalization_version or 0}\n"
            f"before: {item.before_data}\n"
            f"after: {item.proposed_after_data}\n"
            f"signals: {item.source_signal_ids} independent={item.independent_signal_count}\n"
            f"reason: {item.reason}\n"
        )


def show_proposal_normalizations(runtime: EvoMeRuntime) -> None:
    print("\n===== Proposal Policy Normalizations =====")
    items = runtime.list_proposal_normalizations()
    if not items:
        print("No Proposal Policy Normalization records")
        return
    for item in items:
        target = f"RP#{item.normalized_proposal_id}" if item.normalized_proposal_id else "-"
        print(
            f"PN#{item.id} RP#{item.source_proposal_id} -> {target} "
            f"status={item.status} rules={item.rule_count} "
            f"sources={item.source_signal_ids}\n"
            f"reason: {item.reason}\n"
        )


def show_response_policy(runtime: EvoMeRuntime, candidate_id: int) -> None:
    policy = runtime.get_response_policy(candidate_id)
    print(f"\n===== Response Policy CR#{candidate_id} =====")
    print(json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=False))


def show_candidates(runtime: EvoMeRuntime) -> None:
    print("\n===== Candidate Revisions =====")
    items = runtime.list_candidate_revisions()
    if not items:
        print("暂无 Candidate Revision")
        return
    for item in items:
        evaluation = item.evaluation_data or {}
        coverage = float(evaluation.get("source_coverage", evaluation.get("coverage", 0.0)))
        regressions = int(evaluation.get("regression_count", 0))
        method = evaluation.get("method", "legacy")
        evaluation_valid = bool(evaluation.get("evaluation_valid", True))
        evaluation_error = str(evaluation.get("evaluation_error", "") or "")
        eval_note = "" if evaluation_valid else f" evaluator={evaluation_error or 'invalid_output'}"
        stale_reason = str(evaluation.get("stale_reason", "") or "")
        stale_note = f" stale_reason={stale_reason}" if item.status == "stale" and stale_reason else ""
        print(
            f"CR#{item.id} [{item.status}] -> H#{item.hypothesis_id}:r{item.base_revision} "
            f"<- RP#{item.source_proposal_id} eval=v{item.evaluation_version}{eval_note}{stale_note}\n"
            f"local-fit: {item.base_fit_score:.3f} -> {item.candidate_fit_score:.3f} "
            f"gain={item.fit_gain:+.3f} coverage={coverage:.2f} regressions={regressions}\n"
            f"complexity(v{item.complexity_version}): "
            f"{item.base_complexity_score} -> {item.complexity_score} "
            f"delta={item.complexity_delta:+d}\n"
            f"risk={item.risk_level}({item.risk_score:.2f}) "
            f"utility={item.utility_score:+.3f}\n"
            f"WHEN: {item.when_data or {}}\n"
            f"DO: {item.do_data or {}}\n"
            f"EXPECT: {item.expect_data or {}}\n"
        )




def show_candidate_rebases(runtime: EvoMeRuntime) -> None:
    print("\n===== Candidate Rebase Records =====")
    items = runtime.list_candidate_rebases()
    if not items:
        print("暂无 Candidate Rebase Record")
        return
    for item in items:
        new_candidate = f"CR#{item.rebased_candidate_id}" if item.rebased_candidate_id is not None else "-"
        new_proposal = f"RP#{item.rebased_proposal_id}" if item.rebased_proposal_id is not None else "-"
        print(
            f"RB#{item.id} [{item.status}] H#{item.hypothesis_id}: "
            f"CR#{item.source_candidate_id} r{item.from_revision} -> r{item.target_revision} "
            f"proposal={new_proposal} candidate={new_candidate}\n"
            f"signals: {item.source_signal_ids} -> {item.target_signal_ids}\n"
            f"reason: {item.reason}\n"
        )


def show_revision_events(runtime: EvoMeRuntime) -> None:
    print("\n===== Revision Application Events =====")
    items = runtime.list_revision_events()
    if not items:
        print("暂无 Revision Application Event")
        return
    for item in items:
        target = f"r{item.to_revision}" if item.to_revision is not None else "-"
        candidate = f"CR#{item.candidate_id}" if item.candidate_id is not None else "-"
        proposal = f"RP#{item.proposal_id}" if item.proposal_id is not None else "-"
        print(
            f"RE#{item.id} [{item.action}] H#{item.hypothesis_id}: "
            f"r{item.from_revision} -> {target} candidate={candidate} proposal={proposal}\n"
            f"risk={item.risk_level or '-'} utility={item.utility_score:+.3f}\n"
            f"reason: {item.reason}\n"
            f"data: {item.event_data or {}}\n"
        )

def show_interactions(runtime: EvoMeRuntime) -> None:
    print("\n===== Audit Log =====")
    items = runtime.list_interactions()
    if not items:
        print("暂无 InteractionRecord")
    for r in items:
        preview = r.assistant_response[:160]
        suffix = "..." if len(r.assistant_response) > 160 else ""
        eligibility = (r.record_metadata or {}).get("learning_eligibility", {})
        learning_note = (
            f" learning={eligibility.get('category', 'legacy')}"
            f"/memory={bool(eligibility.get('allow_personal_memory', False))}"
            f"/evolution={bool(eligibility.get('allow_evolution', False))}"
            if eligibility
            else ""
        )
        print(
            f"#{r.id} session={r.session_id[:8]} strategy={r.strategy_name}{learning_note}\n"
            f"User: {r.user_input}\nAssistant: {preview}{suffix}\n"
        )


def show_learning_invalidations(runtime: EvoMeRuntime) -> None:
    print("\n===== Learning Invalidations =====")
    items = runtime.list_learning_invalidations()
    if not items:
        print("暂无 Learning Invalidation Event")
        return
    for item in items:
        print(
            f"LI#{item.id} interaction=#{item.interaction_id}\n"
            f"reason: {item.reason}\n"
            f"data: {item.event_data or {}}\n"
        )


def show_skills(runtime: EvoMeRuntime) -> None:
    print("\n===== Skills =====")
    for skill in runtime.skill_registry.list_skills():
        print(f"- {skill.name}: {skill.description}")
        for impl in runtime.skill_registry.list_implementations(skill.name):
            print(f"    -> {impl.name} ({impl.kind})")




def show_token_usage(runtime: EvoMeRuntime) -> None:
    print("\n===== Token Usage (current process) =====")
    summary = runtime.token_usage_summary()
    totals = summary["totals"]
    print(
        f"calls={totals['calls']} prompt={totals['prompt_tokens']} "
        f"completion={totals['completion_tokens']} total={totals['total_tokens']} "
        f"reasoning={totals['reasoning_tokens']} visible={totals['visible_completion_tokens']} "
        f"cached={totals['cached_tokens']} truncated={totals['truncated_calls']}"
    )
    rows = summary.get("by_purpose", [])
    if not rows:
        print("暂无 Token Usage")
        return
    for row in rows:
        print(
            f"- {row['purpose']}: calls={row['calls']} "
            f"prompt={row['prompt_tokens']} completion={row['completion_tokens']} "
            f"total={row['total_tokens']} reasoning={row['reasoning_tokens']} "
            f"visible={row['visible_completion_tokens']} cached={row['cached_tokens']} "
            f"truncated={row['truncated_calls']}"
        )


def show_background_status(runtime: EvoMeRuntime) -> None:
    print("\n===== Background Jobs =====")
    jobs = runtime.list_background_jobs(limit=10)
    if not jobs:
        print("暂无后台任务")
        return
    for job in jobs:
        line = f"{job.id} [{job.status}] kind={job.kind}"
        print(line)
        if job.status == "completed":
            result = job.result
            if job.kind == "revision_analysis":
                print(
                    "  "
                    f"revision_analyses={result.get('revision_analysis_count', 0)} "
                    f"revision_proposals={result.get('revision_proposal_count', 0)} "
                    f"analysis_ids={result.get('analysis_ids', [])} "
                    f"proposal_ids={result.get('proposal_ids', [])}"
                )
            elif job.kind == "candidate_validation":
                print(
                    "  "
                    f"policy_normalizations={result.get('proposal_normalization_count', 0)} "
                    f"normalized_proposals={result.get('normalized_proposal_ids', [])} "
                    f"candidates={result.get('candidate_revision_count', 0)} "
                    f"candidate_ids={result.get('candidate_ids', [])} "
                    f"ready={result.get('ready_to_apply', 0)} "
                    f"review={result.get('needs_review', 0)} "
                    f"rejected={result.get('rejected', 0)} "
                    f"auto_applied={result.get('auto_applied', 0)} "
                    f"new_revisions={result.get('new_revisions', [])}"
                )
            elif job.kind == "revision_auto_apply":
                print(
                    "  "
                    f"auto_applied={result.get('auto_applied', 0)} "
                    f"candidate_ids={result.get('candidate_ids', [])} "
                    f"event_ids={result.get('revision_event_ids', [])} "
                    f"new_revisions={result.get('new_revisions', [])}"
                )
            elif job.kind == "ancestor_revalidation":
                print(
                    "  "
                    f"batches={result.get('revalidation_batches', 0)} "
                    f"hypotheses={result.get('hypothesis_ids', [])} "
                    f"records={result.get('record_ids', [])} "
                    f"rebased_signals={result.get('rebased_signal_ids', [])} "
                    f"resolved={result.get('resolved_signal_ids', [])} "
                    f"superseded={result.get('superseded_signal_ids', [])} "
                    f"uncertain={result.get('uncertain_signal_ids', [])}"
                )
            elif job.kind == "candidate_rebase":
                print(
                    "  "
                    f"source=CR#{result.get('source_candidate_id')} "
                    f"status={result.get('rebase_status')} "
                    f"target=r{result.get('target_revision')} "
                    f"proposal={result.get('rebased_proposal_id')} "
                    f"candidate={result.get('rebased_candidate_id')} "
                    f"candidate_status={result.get('candidate_status')} "
                    f"llm_called={result.get('llm_called', False)}"
                )
            else:
                print(
                    "  "
                    f"interaction={result.get('interaction_id')} "
                    f"learning={result.get('learning_category', 'unknown')} "
                    f"memory_lane={result.get('allow_personal_memory', False)} "
                    f"evolution_lane={result.get('allow_evolution', False)} "
                    f"score={float(result.get('evaluation_score', 0.0)):.2f} "
                    f"experience={result.get('experience_saved', False)} "
                    f"reflection={result.get('reflection_saved', False)} "
                    f"hypothesis={result.get('hypothesis_action', 'skipped')}:"
                    f"{result.get('hypothesis_id')} "
                    f"revision_signals={result.get('revision_signal_count', 0)} "
                    f"revision_analyses={result.get('revision_analysis_count', 0)} "
                    f"revision_proposals={result.get('revision_proposal_count', 0)} "
                    f"policy_normalizations={result.get('proposal_normalization_count', 0)} "
                    f"candidates={result.get('candidate_revision_count', 0)} "
                    f"auto_applied={result.get('revision_auto_apply_count', 0)}"
                )
        elif job.status == "failed":
            print(f"  error={job.error}")

def main() -> None:
    init_db()
    print(f"🌱 EvoMe {APP_VERSION} Core")
    print(
        "命令：/memory  /experience  /reflection  /hypothesis  /revision  "
        "/revision-signals  /revision-pressure  /revision-groups  /ancestor-revalidate  "
        "/signal-revalidations  /evidence-summary  /revision-analyze  "
        "/revision-analyses  /revision-proposals  /candidate-evaluate  /candidates  "
        "/proposal-normalize <id>  /proposal-normalizations  /response-policy <candidate_id>  "
        "/candidate-rebase <id>  /candidate-rebases  "
        "/revision-apply-auto  /candidate-approve <id>  /candidate-reject <id>  "
        "/revision-events  /revision-rollback <hypothesis_id> <revision>  "
        "/learning-invalidate <interaction_id> [reason]  /learning-invalidations  "
        "/status  /token-usage  /audit  /skills  /hooks  /tool-approvals  "
        "/tool-approve <id>  /tool-deny <id>  /exit\n"
    )

    with SessionLocal() as db:
        runtime = EvoMeRuntime(db)
        while True:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            user_input = normalize_cli_input(user_input)

            # Unknown slash-prefixed inputs are operational mistakes, not user
            # conversation. Keep them out of the LLM and all learning paths.
            if user_input.startswith("/"):
                command_name = user_input[1:].split(maxsplit=1)[0].lower()
                if command_name not in COMMAND_NAMES:
                    print(f"未知命令：/{command_name}。输入已忽略，不会进入学习系统。")
                    continue
            if user_input == "/exit":
                print("EvoMe > 正在等待已提交的后台学习任务安全结束...")
                runtime.shutdown(wait=True)
                print("EvoMe > 下次见。")
                break
            if user_input == "/memory":
                show_personal_memories(runtime); continue
            if user_input == "/experience":
                show_experiences(runtime); continue
            if user_input == "/reflection":
                show_reflections(runtime); continue
            if user_input == "/hypothesis":
                show_hypotheses(runtime); continue
            if user_input == "/revision":
                show_revisions(runtime); continue
            if user_input == "/revision-signals":
                show_revision_signals(runtime); continue
            if user_input == "/revision-pressure":
                show_revision_pressure(runtime); continue
            if user_input == "/revision-groups":
                show_revision_groups(runtime); continue
            if user_input == "/ancestor-revalidate":
                job_id = runtime.enqueue_ancestor_revalidation()
                if job_id is None:
                    print("Ancestor Signal Revalidation 已同步执行完成。")
                else:
                    print(f"Ancestor Signal Revalidation 已进入后台队列：{job_id}")
                continue
            if user_input == "/signal-revalidations":
                show_signal_revalidations(runtime); continue
            if user_input == "/evidence-summary":
                show_evidence_summary(runtime); continue
            if user_input == "/revision-analyze":
                job_id = runtime.enqueue_revision_analysis()
                if job_id is None:
                    print("Revision Analyzer 已同步执行完成。")
                else:
                    print(f"Revision Analyzer 已进入后台队列：{job_id}")
                continue
            if user_input == "/revision-analyses":
                show_revision_analyses(runtime); continue
            if user_input == "/revision-proposals":
                show_revision_proposals(runtime); continue
            if user_input.startswith("/proposal-normalize "):
                try:
                    proposal_id = int(user_input.split(maxsplit=1)[1])
                    result = runtime.normalize_proposal(proposal_id)
                    target = f"RP#{result.proposal.id}" if result.proposal is not None else "-"
                    print(
                        f"PN#{result.record.id} RP#{proposal_id} -> {target} "
                        f"status={result.record.status} rules={result.record.rule_count} "
                        f"sources={result.record.source_signal_ids}"
                    )
                except Exception as exc:
                    print(f"Proposal normalization failed: {exc}")
                continue
            if user_input == "/proposal-normalizations":
                show_proposal_normalizations(runtime); continue
            if user_input.startswith("/response-policy "):
                try:
                    candidate_id = int(user_input.split(maxsplit=1)[1])
                    show_response_policy(runtime, candidate_id)
                except Exception as exc:
                    print(f"Response policy display failed: {exc}")
                continue
            if user_input == "/candidate-evaluate":
                job_id = runtime.enqueue_candidate_validation()
                if job_id is None:
                    print("Candidate Evaluator 已同步执行完成。")
                else:
                    print(f"Candidate Evaluator 已进入后台队列：{job_id}")
                continue
            if user_input == "/candidates":
                show_candidates(runtime); continue
            if user_input.startswith("/candidate-rebase "):
                try:
                    candidate_id = int(user_input.split(maxsplit=1)[1])
                    job_id = runtime.enqueue_candidate_rebase(candidate_id)
                    if job_id is None:
                        print(f"Candidate CR#{candidate_id} Rebase 已同步执行完成。")
                    else:
                        print(f"Candidate CR#{candidate_id} Rebase 已进入后台队列：{job_id}")
                except Exception as exc:
                    print(f"❌ Candidate rebase failed: {exc}")
                continue
            if user_input == "/candidate-rebases":
                show_candidate_rebases(runtime); continue
            if user_input == "/revision-apply-auto":
                job_id = runtime.enqueue_revision_auto_apply()
                if job_id is None:
                    print("Revision Auto Apply 已同步执行完成。")
                else:
                    print(f"Revision Auto Apply 已进入后台队列：{job_id}")
                continue
            if user_input.startswith("/candidate-approve "):
                try:
                    candidate_id = int(user_input.split(maxsplit=1)[1])
                    result = runtime.approve_candidate(candidate_id)
                    if result.event is not None:
                        print(
                            f"已批准 CR#{candidate_id}：H#{result.event.hypothesis_id} "
                            f"r{result.event.from_revision} -> r{result.event.to_revision} "
                            f"event=RE#{result.event.id}"
                        )
                except Exception as exc:
                    print(f"❌ Candidate approve failed: {exc}")
                continue
            if user_input.startswith("/candidate-reject "):
                try:
                    candidate_id = int(user_input.split(maxsplit=1)[1])
                    result = runtime.reject_candidate(candidate_id)
                    event_id = result.event.id if result.event is not None else "-"
                    print(f"已拒绝 CR#{candidate_id} event=RE#{event_id}")
                except Exception as exc:
                    print(f"❌ Candidate reject failed: {exc}")
                continue
            if user_input == "/revision-events":
                show_revision_events(runtime); continue
            if user_input.startswith("/revision-rollback "):
                try:
                    parts = user_input.split()
                    if len(parts) != 3:
                        raise ValueError("用法：/revision-rollback <hypothesis_id> <revision>")
                    hypothesis_id = int(parts[1])
                    target_revision = int(parts[2])
                    result = runtime.rollback_revision(hypothesis_id, target_revision)
                    if result.event is not None:
                        print(
                            f"已回滚 H#{hypothesis_id}：r{result.event.from_revision} "
                            f"-> r{result.event.to_revision} event=RE#{result.event.id}"
                        )
                except Exception as exc:
                    print(f"❌ Revision rollback failed: {exc}")
                continue
            if user_input.startswith("/learning-invalidate "):
                try:
                    parts = user_input.split(maxsplit=2)
                    if len(parts) < 2:
                        raise ValueError("用法：/learning-invalidate <interaction_id> [reason]")
                    interaction_id = int(parts[1])
                    reason = parts[2].strip() if len(parts) >= 3 else "operator_marked_non_learning"
                    event = runtime.invalidate_learning_from_interaction(
                        interaction_id, reason=reason
                    )
                    review = (event.event_data or {}).get(
                        "applied_candidate_ids_requiring_manual_review", []
                    )
                    print(
                        f"已失效 Interaction #{interaction_id} 的派生学习：LI#{event.id} "
                        f"signals={(event.event_data or {}).get('revision_signal_ids', [])} "
                        f"stale_candidates={(event.event_data or {}).get('stale_candidate_ids', [])}"
                    )
                    if review:
                        print(
                            "⚠️ 已有正式 Revision 使用了受污染 Candidate，未自动回滚。"
                            f"需要人工检查 Candidate：{review}"
                        )
                except Exception as exc:
                    print(f"❌ Learning invalidation failed: {exc}")
                continue
            if user_input == "/learning-invalidations":
                show_learning_invalidations(runtime); continue
            if user_input == "/status":
                show_background_status(runtime); continue
            if user_input == "/token-usage":
                show_token_usage(runtime); continue
            if user_input == "/audit":
                show_interactions(runtime); continue
            if user_input == "/hooks":
                events = runtime.list_hook_events(limit=20)
                if not events:
                    print("暂无 Hook 事件（或 S04 Hooks 已关闭）")
                for entry in events:
                    print(
                        f"{entry.timestamp} {entry.event} "
                        f"[{entry.outcome}] tool={entry.tool_name or '-'} "
                        f"risk={entry.risk or '-'} status={entry.status or '-'}"
                        + (f" reason={entry.reason}" if entry.reason else "")
                    )
                continue
            if user_input == "/tool-approvals":
                items = runtime.list_pending_tool_approvals()
                if not items:
                    print("暂无待确认 Tool 调用")
                for item in items:
                    print(
                        f"{item.id} [{item.risk}] tool={item.tool_name} "
                        f"args={item.arguments} reason={item.reason}"
                    )
                continue
            if user_input.startswith("/tool-approve "):
                approval_id = user_input.split(maxsplit=1)[1].strip()
                result = runtime.approve_tool_call(approval_id)
                if result is None:
                    print("Agent Loop 未启用。")
                elif result.success:
                    print(f"✅ Tool 已执行：{result.tool_name}\n{result.output}")
                else:
                    print(f"❌ Tool 执行失败：{result.error}")
                continue
            if user_input.startswith("/tool-deny "):
                approval_id = user_input.split(maxsplit=1)[1].strip()
                ok = runtime.deny_tool_call(approval_id)
                print("已拒绝。" if ok else "未找到该待确认请求。")
                continue
            if user_input == "/skills":
                show_skills(runtime); continue

            if user_input.startswith("/"):
                print(f"命令格式不正确：{user_input}。该输入不会发送给 LLM，也不会进入学习系统。")
                continue

            try:
                print("\nEvoMe > ", end="", flush=True)
                result = runtime.stream_run(
                    user_input,
                    on_token=lambda token: print(token, end="", flush=True),
                )
                print("\n")
                print(
                    "FAST | "
                    f"strategy={result['strategy_name']} "
                    f"memory={result['relevant_memory_count']} "
                    f"background_job={result['background_job_id']} "
                    f"status={result['background_status']}\n"
                )
            except Exception as exc:
                print(f"\n❌ EvoMe Error: {exc}\n")


if __name__ == "__main__":
    main()
