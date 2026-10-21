from optbench.registry import PROBLEMS, list_problems


def test_all_30_specification_ids_are_registered() -> None:
    expected = {
        *(f"D{i:02d}" for i in range(1, 6)),
        *(f"V{i:02d}" for i in range(1, 6)),
        *(f"N{i:02d}" for i in range(1, 6)),
        *(f"S{i:02d}" for i in range(1, 6)),
        *(f"C{i:02d}" for i in range(1, 6)),
        *(f"B{i:02d}" for i in range(1, 6)),
    }
    assert set(PROBLEMS) == expected
    assert len(list_problems()) == 30


def test_listing_does_not_construct_optional_backends() -> None:
    records = list_problems()
    assert records[0]["problem_id"] == "B01"
    assert all("factory" not in record for record in records)
