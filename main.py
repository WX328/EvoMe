from app.database import Base, engine, SessionLocal
from app.runtime import EvoMeRuntime


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


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
        print(
            f"#{e.id} [{e.task_type}] score={e.score:.2f} success={e.success}\n"
            f"{e.result_summary}\n"
        )


def show_reflections(runtime: EvoMeRuntime) -> None:
    print("\n===== Reflection =====")
    items = runtime.list_reflections()
    if not items:
        print("暂无 Reflection")
    for r in items:
        print(
            f"#{r.id} [{r.reflection_type}] confidence={r.confidence:.2f}\n"
            f"观察: {r.observation}\n经验: {r.lesson}\n"
            f"Strategy Signal: {r.strategy_signal}\n"
            f"Skill Signal: {r.skill_signal}\n"
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


def main() -> None:
    init_db()
    print("🌱 EvoMe V0.1.1 Core")
    print("命令：/memory  /experience  /reflection  /audit  /skills  /exit\n")

    with SessionLocal() as db:
        runtime = EvoMeRuntime(db)
        while True:
            user_input = input("You > ").strip()
            if not user_input:
                continue
            if user_input == "/exit":
                print("EvoMe > 下次见。")
                break
            if user_input == "/memory":
                show_personal_memories(runtime); continue
            if user_input == "/experience":
                show_experiences(runtime); continue
            if user_input == "/reflection":
                show_reflections(runtime); continue
            if user_input == "/audit":
                show_interactions(runtime); continue
            if user_input == "/skills":
                show_skills(runtime); continue

            try:
                result = runtime.run(user_input)
                print(f"\nEvoMe > {result['response']}\n")
                print(
                    "DEBUG | "
                    f"strategy={result['strategy_name']} "
                    f"score={result['evaluation_score']:.2f} "
                    f"interaction={result['interaction_id']} "
                    f"personal_memory+={result['personal_memories_saved']} "
                    f"experience={result['experience_saved']} "
                    f"reflection={result['reflection_saved']}\n"
                )
            except Exception as exc:
                print(f"\n❌ EvoMe Error: {exc}\n")


if __name__ == "__main__":
    main()
