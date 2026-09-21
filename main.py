from app.database import Base, engine, SessionLocal
from app.migrations import migrate_v0_2, migrate_v0_2_1
from app.runtime import EvoMeRuntime


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
        print(
            f"H#{h.id} [{h.status}] confidence={h.confidence:.2f} revision={h.current_revision}\n"
            f"{h.title}\n"
            f"WHEN: {h.applicable_context or {}}\n"
            f"DO: {h.proposed_change}\n"
            f"EXPECT: {h.expected_effect}\n"
            f"Evidence: support={h.supporting_count} refine={h.refinement_count} "
            f"counter={h.counter_count} independent={h.independent_evidence_count} "
            f"independent_support={h.independent_support_count} "
            f"diversity={h.context_diversity}\n"
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
            f"confidence={signal.confidence:.2f}\n"
            f"{signal.dimension} / {signal.operation} / {signal.target_field}\n"
            f"before: {signal.before_data}\n"
            f"observed: {signal.observed_data}\n"
            f"change: {signal.canonical_change}\n"
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
            f"Revision Groups: {len(summary['revision_groups'])} "
            f"analysis_candidates={len(summary['analysis_candidates'])}\n"
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
            f"confidence={item.direction_confidence:.2f}\n"
            f"before: {item.before_data}\n"
            f"after: {item.proposed_after_data}\n"
            f"signals: {item.source_signal_ids} independent={item.independent_signal_count}\n"
            f"reason: {item.reason}\n"
        )


def show_interactions(runtime: EvoMeRuntime) -> None:
    print("\n===== Audit Log =====")
    items = runtime.list_interactions()
    if not items:
        print("暂无 InteractionRecord")
    for r in items:
        preview = r.assistant_response[:160]
        suffix = "..." if len(r.assistant_response) > 160 else ""
        print(
            f"#{r.id} session={r.session_id[:8]} strategy={r.strategy_name}\n"
            f"User: {r.user_input}\nAssistant: {preview}{suffix}\n"
        )


def show_skills(runtime: EvoMeRuntime) -> None:
    print("\n===== Skills =====")
    for skill in runtime.skill_registry.list_skills():
        print(f"- {skill.name}: {skill.description}")
        for impl in runtime.skill_registry.list_implementations(skill.name):
            print(f"    -> {impl.name} ({impl.kind})")




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
            else:
                print(
                    "  "
                    f"interaction={result.get('interaction_id')} "
                    f"score={float(result.get('evaluation_score', 0.0)):.2f} "
                    f"experience={result.get('experience_saved', False)} "
                    f"reflection={result.get('reflection_saved', False)} "
                    f"hypothesis={result.get('hypothesis_action', 'skipped')}:"
                    f"{result.get('hypothesis_id')} "
                    f"revision_signals={result.get('revision_signal_count', 0)} "
                    f"revision_analyses={result.get('revision_analysis_count', 0)} "
                    f"revision_proposals={result.get('revision_proposal_count', 0)}"
                )
        elif job.status == "failed":
            print(f"  error={job.error}")

def main() -> None:
    init_db()
    print("🌱 EvoMe V0.2.1-dev Core")
    print(
        "命令：/memory  /experience  /reflection  /hypothesis  /revision  "
        "/revision-signals  /revision-groups  /evidence-summary  /revision-analyze  "
        "/revision-analyses  /revision-proposals  /status  /audit  /skills  /exit\n"
    )

    with SessionLocal() as db:
        runtime = EvoMeRuntime(db)
        while True:
            user_input = input("You > ").strip()
            if not user_input:
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
            if user_input == "/revision-groups":
                show_revision_groups(runtime); continue
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
            if user_input == "/status":
                show_background_status(runtime); continue
            if user_input == "/audit":
                show_interactions(runtime); continue
            if user_input == "/skills":
                show_skills(runtime); continue

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
