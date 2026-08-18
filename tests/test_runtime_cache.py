from medchange.runtime.cache import (
    ResultCache,
)


def test_result_cache():
    cache = ResultCache(
        max_entries=2
    )

    cache.set(
        "a",
        {
            "value": 1
        },
    )

    assert (
        cache.get(
            "a"
        )[
            "value"
        ]
        == 1
    )


def test_lru_eviction():
    cache = ResultCache(
        max_entries=2
    )

    cache.set(
        "a",
        {
            "value": 1
        },
    )

    cache.set(
        "b",
        {
            "value": 2
        },
    )

    cache.set(
        "c",
        {
            "value": 3
        },
    )

    assert (
        cache.get(
            "a"
        )
        is None
    )

    assert len(
        cache
    ) == 2