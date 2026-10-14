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
        "/tool-approve <id>  /tool-deny <id>  /plans  /plan <goal>  "
        "/plan-show <id>  /plan-run <id>  /plan-replan <id>  "
        "/plan-cancel <id>  /subagent <read-only task>  "
        "/skill-docs  /skill-load <name>  /context-stats  "
        "/agent-memory-list  /agent-memory-find <query>  "
        "/agent-memory-propose <user statement>  /agent-memory-pending  "
        "/agent-memory-approve <id>  /agent-memory-deny <id>  "
        "/agent-memory-preview  /agent-memory-consolidate  "
        "/task-list  /task-ready  /task-show <id>  /task-create <title>  "
        "/task-depend <id> <dependency>  /task-start <id> [owner]  "
        "/task-done <id> [owner]  /task-release <id> [owner]  "
        "/task-bg-run <id>  /task-bg-list  /task-bg-show <id>  "
        "/task-bg-wait <id> [seconds]  /task-bg-recover <id>  "
        "/cron-at <ISO> | <只读任务>  /cron-add <expr> <offset> <runs> | <只读任务>  "
        "/cron-after <seconds> | <只读任务>  /cron-list  /cron-show <id>  "
        "/cron-pause <id>  /cron-resume <id>  /cron-cancel <id>  "
        "/cron-events  /cron-tick  "
        "/team-propose alice:role bob:role  /team-approve <proposal_id>  "
        "/team-reject <proposal_id>  /team-resume <team_id>  /team-status  "
        "/team-bind <task_id> <member|any> [worktree]  "
        "/team-send <member> | <message>  /team-inbox  "
        "/team-plan-request <member> | <plan>  "
        "/team-plan-approve <member>  /team-plan-reject <member>  "
        "/team-stop <member>  /team-recover <task_id>  /team-shutdown  "
        "/mcp-servers  /mcp-connect <server>  /mcp-tools  "
        "/mcp-call <mcp__server__tool> <JSON>  "
        "/mcp-disconnect <server>  /exit\n"
    )

    with SessionLocal() as db:
        runtime = EvoMeRuntime(
            db, enable_scheduler=True, enable_teams=True, enable_mcp=True,
        )
        while True:
            user_input = input("You > ").strip()
            # Deliver completed team messages on the next foreground turn.
            # Operator commands never forward them into the learning path.
            try:
                team_events = runtime.poll_team_events()
                for item in team_events:
                    print(
                        f"[TEAM {item['from']}:{item['type']}] "
                        f"{str(item['content'])[:500]}"
                    )
            except Exception as exc:
                print(f"S13 inbox warning: {exc}")
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
            if user_input == "/agent-memory-list":
                try:
                    items = runtime.list_agent_memories()
                    if not items:
                        print("S09 尚无 Agent 工作记忆（/memory 是独立的 SQLite Personal Memory）")
                    for item in items:
                        print(f"  {item['name']} [{item['type']}] {item['description']}")
                except Exception as exc:
                    print(f"S09 Memory Error: {exc}")
                continue
            if user_input.startswith("/agent-memory-find "):
                try:
                    query = user_input.split(maxsplit=1)[1].strip()
                    result = runtime.recall_agent_memories(query)
                    if not result.success:
                        print(f"S09 Memory Error: {result.error}")
                    else:
                        payload = result.output
                        print(f"S09 selected {payload['selected']} records")
                        for item in payload["memories"]:
                            print(f"  [{item['type']}] {item['name']}: {item['body']}"
                                  + (" (truncated)" if item['truncated'] else ""))
                except Exception as exc:
                    print(f"S09 Memory Error: {exc}")
                continue
            if user_input.startswith("/agent-memory-propose "):
                try:
                    utterance = user_input.split(maxsplit=1)[1].strip()
                    response = runtime.propose_agent_memories(utterance)
                    if not response["pending"]:
                        print(f"无可审核候选：{response.get('reason', 'no durable evidence')}")
                    for item in response["pending"]:
                        print(
                            f"  候选 {item['id']} [{item['type']}] {item['name']} "
                            f"{item['description']} | 原话：{item['source_quote']}"
                        )
                    if response["pending"]:
                        print("注意：候选尚未写入磁盘，确认后使用 /agent-memory-approve <id>")
                except Exception as exc:
                    print(f"S09 Memory Error: {exc}")
                continue
            if user_input == "/agent-memory-pending":
                try:
                    for item in runtime.pending_agent_memories():
                        print(
                            f"  {item.id} [{item.memory_type}] {item.name}: "
                            f"{item.description} | {item.source_quote}"
                        )
                except Exception as exc:
                    print(f"S09 Memory Error: {exc}")
                continue
            if user_input.startswith("/agent-memory-approve "):
                try:
                    candidate_id = user_input.split(maxsplit=1)[1].strip()
                    item = runtime.approve_agent_memory(candidate_id)
                    print(f"S09 已保存：{item.name} [{item.memory_type}] -> {item.relative_path}")
                except Exception as exc:
                    print(f"S09 Memory Error: {exc}")
                continue
            if user_input.startswith("/agent-memory-deny "):
                try:
                    candidate_id = user_input.split(maxsplit=1)[1].strip()
                    print("S09 已拒绝" if runtime.deny_agent_memory(candidate_id) else "未找到候选")
                except Exception as exc:
                    print(f"S09 Memory Error: {exc}")
                continue
            if user_input == "/agent-memory-preview":
                try:
                    print(runtime.preview_agent_memory_consolidation())
                except Exception as exc:
                    print(f"S09 Memory Error: {exc}")
                continue
            if user_input == "/agent-memory-consolidate":
                try:
                    print(runtime.consolidate_agent_memories())
                except Exception as exc:
                    print(f"S09 Memory Error: {exc}")
                continue
            if user_input in {"/task-list", "/task-ready"}:
                try:
                    tasks = (
                        runtime.list_durable_tasks()
                        if user_input == "/task-list"
                        else runtime.ready_durable_tasks()
                    )
                    if not tasks:
                        print("S10 暂无符合条件的持久化任务")
                    for task in tasks:
                        blockers = ",".join(task["blockedByOpen"]) or "-"
                        print(
                            f"  {task['id']} [{task['status']}] "
                            f"ready={task['ready']} blockedByOpen={blockers} "
                            f"owner={task['owner'] or '-'} | {task['title']}"
                        )
                except Exception as exc:
                    print(f"S10 Task Error: {exc}")
                continue
            if user_input.startswith("/task-show "):
                try:
                    task_id = user_input.split(maxsplit=1)[1].strip()
                    print(runtime.get_durable_task(task_id))
                except Exception as exc:
                    print(f"S10 Task Error: {exc}")
                continue
            if user_input.startswith("/task-create "):
                try:
                    title = user_input.split(maxsplit=1)[1].strip()
                    task = runtime.create_durable_task(title)
                    print(f"S10 已创建 {task['id']} [pending] {task['title']}")
                except Exception as exc:
                    print(f"S10 Task Error: {exc}")
                continue
            if user_input.startswith("/task-depend "):
                try:
                    _, task_id, dependency_id = user_input.split()
                    task = runtime.add_durable_dependency(task_id, dependency_id)
                    print(f"S10 {task['id']} blockedBy={task['blockedBy']}")
                except Exception as exc:
                    print(f"S10 Task Error: {exc}")
                continue
            if user_input.startswith(("/task-start ", "/task-done ", "/task-release ")):
                try:
                    tokens = user_input.split()
                    if not 2 <= len(tokens) <= 3:
                        raise ValueError("用法：/task-start|task-done|task-release <id> [owner]")
                    action, task_id = tokens[:2]
                    owner = tokens[2] if len(tokens) == 3 else "operator"
                    if action == "/task-start":
                        task = runtime.claim_durable_task(task_id, owner)
                    elif action == "/task-done":
                        task = runtime.complete_durable_task(task_id, owner)
                    else:
                        task = runtime.release_durable_task(task_id, owner)
                    print(f"S10 {task['id']} [{task['status']}] owner={task['owner'] or '-'}")
                except Exception as exc:
                    print(f"S10 Task Error: {exc}")
                continue
            if user_input in {"/cron-list", "/cron-events", "/cron-tick"}:
                try:
                    if user_input == "/cron-list":
                        rows = runtime.list_cron_schedules()
                    elif user_input == "/cron-events":
                        rows = runtime.cron_events()
                    else:
                        rows = runtime.poll_cron_schedules()
                    print("S12 尚无记录" if not rows else "")
                    for item in rows:
                        print(item)
                except Exception as exc:
                    print(f"S12 Cron Error: {exc}")
                continue
            if user_input.startswith(("/cron-at ", "/cron-add ", "/cron-after ")):
                try:
                    import shlex
                    head, sep, title = user_input.partition("|")
                    if not sep or not title.strip():
                        raise ValueError("请使用 | 分隔时间参数与只读任务名称")
                    pieces = shlex.split(head)
                    mode = pieces[0]
                    if mode == "/cron-at" and len(pieces) == 2:
                        row = runtime.create_cron_at(at=pieces[1], title=title.strip())
                    elif mode == "/cron-add" and len(pieces) == 4:
                        row = runtime.create_cron_recurring(
                            expression=pieces[1], utc_offset=pieces[2],
                            max_runs=int(pieces[3]), title=title.strip(),
                        )
                    elif mode == "/cron-after" and len(pieces) == 2:
                        row = runtime.create_cron_session(
                            seconds=float(pieces[1]), title=title.strip(),
                        )
                    else:
                        raise ValueError("命令格式错误；参考启动帮助中的 cron 示例")
                    print(
                        f"S12 已创建 {row['id']} [{row['kind']}] "
                        f"next={row['next_run_at']} status={row['status']}"
                    )
                except Exception as exc:
                    print(f"S12 Cron Error: {exc}")
                continue
            if user_input.startswith(("/cron-show ", "/cron-pause ", "/cron-resume ", "/cron-cancel ")):
                try:
                    command, schedule_id = user_input.split(maxsplit=1)
                    schedule_id = schedule_id.strip()
                    if command == "/cron-show":
                        row = runtime.show_cron_schedule(schedule_id)
                    elif command == "/cron-pause":
                        row = runtime.pause_cron_schedule(schedule_id)
                    elif command == "/cron-resume":
                        row = runtime.resume_cron_schedule(schedule_id)
                    else:
                        row = runtime.cancel_cron_schedule(schedule_id)
                    print(f"S12 {row}")
                except Exception as exc:
                    print(f"S12 Cron Error: {exc}")
                continue
            if user_input == "/team-status":
                try:
                    print(runtime.team_status())
                except Exception as exc:
                    print(f"S13 Team Error: {exc}")
                continue
            if user_input == "/team-inbox":
                try:
                    for item in runtime.team_inbox():
                        print(
                            f"  {item['from']} -> {item['to']} [{item['type']}] "
                            f"{str(item['content'])[:1200]}"
                        )
                except Exception as exc:
                    print(f"S13 Team Error: {exc}")
                continue
            if user_input == "/team-shutdown":
                try:
                    runtime.team_shutdown()
                    print("S13 team shutdown requested; roster remains on disk")
                except Exception as exc:
                    print(f"S13 Team Error: {exc}")
                continue
            if user_input.startswith("/team-propose "):
                try:
                    pairs = []
                    for token in user_input.split()[1:]:
                        name, sep, role = token.partition(":")
                        if not sep:
                            raise ValueError("format: /team-propose alice:config bob:tests")
                        pairs.append((name, role))
                    print("S13 proposal:", runtime.team_propose(pairs))
                    print("Review roles, then explicitly /team-approve <proposal_id>")
                except Exception as exc:
                    print(f"S13 Team Error: {exc}")
                continue
            if user_input.startswith((
                "/team-approve ", "/team-reject ", "/team-resume ",
                "/team-stop ", "/team-recover ",
                "/team-plan-approve ", "/team-plan-reject ",
            )):
                try:
                    command, argument = user_input.split(maxsplit=1)
                    key = argument.strip()
                    if command == "/team-approve":
                        output = runtime.team_approve(key)
                    elif command == "/team-reject":
                        output = runtime.team_reject(key)
                    elif command == "/team-resume":
                        output = runtime.team_resume(key)
                    elif command == "/team-stop":
                        output = runtime.team_stop_member(key)
                    elif command == "/team-recover":
                        output = runtime.team_recover(key)
                    else:
                        output = runtime.team_plan_respond(
                            key, approved=command == "/team-plan-approve"
                        )
                    print(f"S13 {output}")
                except Exception as exc:
                    print(f"S13 Team Error: {exc}")
                continue
            if user_input.startswith("/team-bind "):
                try:
                    fields = user_input.split()
                    if len(fields) not in {3, 4}:
                        raise ValueError("usage: /team-bind <task_id> <member|any> [worktree_slug]")
                    print(runtime.team_bind(
                        fields[1], fields[2],
                        fields[3] if len(fields) == 4 else "",
                    ))
                except Exception as exc:
                    print(f"S13 Team Error: {exc}")
                continue
            if user_input.startswith(("/team-send ", "/team-plan-request ")):
                try:
                    head, separator, text = user_input.partition("|")
                    if not separator or not text.strip():
                        raise ValueError("use | to separate teammate and text")
                    command, member = head.strip().split()
                    output = (
                        runtime.team_message(member, text.strip())
                        if command == "/team-send"
                        else runtime.team_plan_request(member, text.strip())
                    )
                    print(f"S13 {output}")
                except Exception as exc:
                    print(f"S13 Team Error: {exc}")
                continue
            if user_input == "/mcp-servers":
                try:
                    print("S14 configured servers:")
                    for entry in runtime.list_mcp_servers():
                        print(" ", entry)
                except Exception as exc:
                    print(f"S14 MCP Error: {exc}")
                continue
            if user_input == "/mcp-tools":
                try:
                    for item in runtime.list_mcp_tools():
                        print(f"  {item['tool']} : {item['description'][:120]}")
                except Exception as exc:
                    print(f"S14 MCP Error: {exc}")
                continue
            if user_input.startswith("/mcp-connect "):
                try:
                    server = user_input.split(maxsplit=1)[1].strip()
                    result = runtime.connect_mcp_server(server)
                    print(f"S14 MCP connected {server}: {result}")
                    print("External calls require /tool-approve for each invocation.")
                except Exception as exc:
                    print(f"S14 MCP Error: {exc}")
                continue
            if user_input.startswith("/mcp-disconnect "):
                try:
                    server = user_input.split(maxsplit=1)[1].strip()
                    runtime.disconnect_mcp_server(server)
                    print(f"S14 MCP disconnected: {server}")
                except Exception as exc:
                    print(f"S14 MCP Error: {exc}")
                continue
            if user_input.startswith("/mcp-call "):
                try:
                    import json
                    parts = user_input.split(maxsplit=2)
                    if len(parts) != 3:
                        raise ValueError(
                            'usage: /mcp-call mcp__server__tool {"key":"value"}'
                        )
                    arguments = json.loads(parts[2])
                    if not isinstance(arguments, dict):
                        raise ValueError("MCP tool parameters must be a JSON object")
                    result = runtime.invoke_mcp_tool(parts[1], arguments)
                    if result.status == "approval_required":
                        print(
                            f"S14 MCP approval required: {result.approval_id}; "
                            f"inspect /tool-approvals then /tool-approve {result.approval_id}"
                        )
                    elif result.success:
                        print(f"S14 MCP result: {result.output}")
                    else:
                        print(f"S14 MCP error: {result.error}")
                except Exception as exc:
                    print(f"S14 MCP Error: {exc}")
                continue
            if user_input == "/task-bg-list":
                try:
                    items = runtime.list_background_tasks()
                    if not items:
                        print("S11 暂无后台任务")
                    for item in items:
                        print(
                            f"  {item['task_id']} [{item['task_status']}] "
                            f"job={item['job_id'] or '-'} "
                            f"job_status={item['job_status']} | {item['title']}"
                        )
                except Exception as exc:
                    print(f"S11 Background Error: {exc}")
                continue
            if user_input.startswith("/task-bg-run "):
                try:
                    task_id = user_input.split(maxsplit=1)[1].strip()
                    result = runtime.enqueue_background_task(task_id)
                    print(
                        f"S11 已提交：{result['task_id']} job={result['job_id']}; "
                        "只读研究已进入后台队列。通过 /task-bg-show 查看结果。"
                    )
                except Exception as exc:
                    print(f"S11 Background Error: {exc}")
                continue
            if user_input.startswith("/task-bg-show "):
                try:
                    task_id = user_input.split(maxsplit=1)[1].strip()
                    state = runtime.background_task_status(task_id)
                    print(
                        f"S11 {task_id} task={state['task_status']} "
                        f"job={state['job_status']} owner={state['owner'] or '-'}"
                    )
                    if state.get("summary"):
                        print(f"  摘要：{state['summary']}")
                        for row in state.get("evidence", []):
                            print(f"    · {row.get('tool')} {row.get('path')}")
                    elif state.get("result"):
                        print(f"  持久化结果：{state['result']}")
                    if state.get("error"):
                        print(f"  error={state['error']}")
                except Exception as exc:
                    print(f"S11 Background Error: {exc}")
                continue
            if user_input.startswith("/task-bg-wait "):
                try:
                    parts = user_input.split()
                    if len(parts) not in {2, 3}:
                        raise ValueError("用法：/task-bg-wait <task_id> [seconds]")
                    seconds = float(parts[2]) if len(parts) == 3 else 30.0
                    if not 0 <= seconds <= 120:
                        raise ValueError("timeout must be 0-120 seconds")
                    state = runtime.wait_for_durable_background_task(
                        parts[1], timeout=seconds,
                    )
                    print(
                        f"S11 {parts[1]} task={state['task_status']} "
                        f"job={state['job_status']}"
                    )
                except Exception as exc:
                    print(f"S11 Background Error: {exc}")
                continue
            if user_input.startswith("/task-bg-recover "):
                try:
                    task_id = user_input.split(maxsplit=1)[1].strip()
                    task = runtime.recover_background_task(task_id)
                    print(
                        f"S11 已手动恢复 {task_id} -> {task['status']}; "
                        "不会自动执行，请检查后再次使用 /task-bg-run"
                    )
                except Exception as exc:
                    print(f"S11 Background Error: {exc}")
                continue
            if user_input == "/context-stats":
                print(runtime.context_compaction_stats())
                continue
            if user_input == "/skill-docs":
                try:
                    catalog = runtime.list_instruction_skills()
                    if not catalog:
                        print("没有找到有效 SKILL.md。请检查 skills/<name>/SKILL.md。")
                    for item in catalog:
                        print(f"  {item['name']} — {item['description']}")
                except Exception as exc:
                    print(f"❌ Skill Loading Error: {exc}")
                continue
            if user_input == "/skill-load":
                print("用法：/skill-load <name>")
                continue
            if user_input.startswith("/skill-load "):
                try:
                    name = user_input.split(maxsplit=1)[1].strip()
                    result = runtime.load_instruction_skill(name)
                    if result.success:
                        output = result.output
                        print(
                            f"✅ Loaded skill: {output['name']}\n"
                            f"  来源：{output['relative_path']}\n\n"
                            f"{output['content']}"
                        )
                    else:
                        print(f"❌ Skill Loading Error: {result.error}")
                except Exception as exc:
                    print(f"❌ Skill Loading Error: {exc}")
                continue
            if user_input == "/subagent":
                print("用法：/subagent <只读研究任务>")
                continue
            if user_input.startswith("/subagent "):
                try:
                    task = user_input.split(maxsplit=1)[1].strip()
                    result = runtime.run_research_subagent(task)
                    if not result.success:
                        print(f"❌ Research Subagent 未完成：{result.error}")
                    else:
                        output = result.output
                        print(
                            f"✅ Research Subagent completed | "
                            f"tool_calls={output['tool_call_count']}\n"
                            f"{output['summary']}"
                        )
                        for row in output["evidence"]:
                            lines = (
                                f" [{row['start_line']}:{row['end_line']}]"
                                if "start_line" in row and "end_line" in row else ""
                            )
                            print(
                                f"  · {row['tool']} {row['path']}{lines}"
                                + ("（截断）" if row.get("truncated") else "")
                            )
                except Exception as exc:
                    print(f"❌ Research Subagent Error: {exc}")
                continue
            if user_input == "/plans":
                try:
                    items = runtime.list_plans()
                    if not items:
                        print("暂无执行计划")
                    for plan in items:
                        print(
                            f"{plan.id} [{plan.status}] "
                            f"{plan.completed_count}/{len(plan.steps)} "
                            f"{plan.goal[:120]}"
                        )
                except Exception as exc:
                    print(f"❌ Planning Error: {exc}")
                continue
            if user_input.startswith((
                "/plan ", "/plan-show ", "/plan-run ",
                "/plan-replan ", "/plan-cancel ",
            )):
                try:
                    command, value = user_input.split(maxsplit=1)
                    value = value.strip()
                    if not value:
                        raise ValueError("请提供目标或计划 ID")
                    if command == "/plan":
                        plan = runtime.create_plan(value)
                    elif command == "/plan-show":
                        plan = runtime.get_plan(value)
                    elif command == "/plan-run":
                        plan = runtime.resume_plan(value)
                    elif command == "/plan-replan":
                        plan = runtime.replan(value)
                    else:
                        plan = runtime.cancel_plan(value)
                    print(runtime.describe_plan(plan.id, detail=command == "/plan-show"))
                except Exception as exc:
                    print(f"❌ Planning Error: {exc}")
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
