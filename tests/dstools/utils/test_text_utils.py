import pytest
from dstools.utils.text import height_str_to_cm, time_str_to_seconds


class TestHeightStrToCm:
    def test_happy(self):
        test_data = ["5'9\"", "7'", "6-3", '5"']
        digits = 2
        expected_results = [175.26, 213.36, 190.5, 12.7]
        for d, r in zip(test_data, expected_results):
            assert height_str_to_cm(d, digits) == r

    def test_bad_inputs(self):
        test_data = ["5", "5\"7'", "6-3-4", "5-"]
        digits = 2
        exceptions = [ValueError, ValueError, ValueError, ValueError]
        for d, ex in zip(test_data, exceptions):
            with pytest.raises(ex):
                height_str_to_cm(d, digits)


class TestTimeStrToSeconds:
    def test_happy(self):
        test_data = ["1:00:00", "1:00", "1", "67"]
        expected_results = [3600, 60, 1, 67]
        for d, r in zip(test_data, expected_results):
            assert time_str_to_seconds(d) == r

    def test_bad_inputs(self):
        test_data = ["1:00:00:00", "Test", -1]
        exceptions = [ValueError, ValueError, AttributeError]
        for d, ex in zip(test_data, exceptions):
            with pytest.raises(ex):
                time_str_to_seconds(d)
