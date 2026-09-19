from app.database import Base, engine, SessionLocal
from app.runtime import EvoMeRuntime


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def show_memories(runtime: EvoMeRuntime) -> None:
    memories = runtime.list_memories()
    print("\n===== Memory =====")
    if not memories:
        print("暂无 Memory")
    for m in memories:
        print(
            f"#{m.id} [{m.memory_type}] "
            f"importance={m.importance:.2f} confidence={m.confidence:.2f}\n"
            f"{m.content}\n"
        )


def show_skills(runtime: EvoMeRuntime) -> None:
    print("\n===== Skills =====")
    for skill in runtime.skill_registry.list_skills():
        impls = runtime.skill_registry.list_implementations(skill.name)
        print(f"- {skill.name}: {skill.description}")
        for impl in impls:
            print(f"    -> {impl.name} ({impl.kind})")


def main() -> None:
    init_db()
    print("🌱 EvoMe V0.1 Core")
    print("命令：/memory  /skills  /exit\n")

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
                show_memories(runtime)
                continue
            if user_input == "/skills":
                show_skills(runtime)
                continue

            try:
                result = runtime.run(user_input)
                print(f"\nEvoMe > {result['response']}\n")
                print(
                    "DEBUG | "
                    f"strategy={result['strategy_name']} "
                    f"score={result['evaluation_score']:.2f} "
                    f"memory_saved={result['memory_saved']}\n"
                )
            except Exception as exc:
                print(f"\n❌ EvoMe Error: {exc}\n")


if __name__ == "__main__":
    main()
