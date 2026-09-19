from app.memory.service import MemoryService


def test_save_and_list_memory(db):
    service = MemoryService(db)

    service.save(
        memory_type="experience",
        content="EvoMe 第一次拥有长期 Memory",
        importance=0.9,
    )

    memories = service.list_all()

    assert len(memories) == 1
    assert "长期 Memory" in memories[0].content


def test_memory_search(db):
    service = MemoryService(db)

    service.save(
        memory_type="experience",
        content="Python 依赖冲突时先检查虚拟环境和版本约束",
        importance=0.9,
    )
    service.save(
        memory_type="experience",
        content="今天阅读了一篇 Agent 论文",
        importance=0.5,
    )

    results = service.search(
        "Python 依赖冲突怎么办",
        top_k=3,
    )

    assert results
    assert "Python" in results[0].content
