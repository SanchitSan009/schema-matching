"""Small checkpoint benchmark: formatting equivalence and meaning preservation."""

import unittest

from normalize import normalize_column


class NormalizationTests(unittest.TestCase):
    def test_formatting_benchmark(self):
        cases = {
            "InputVoltage": "input voltage",
            "inputVoltage": "input voltage",
            "input_voltage": "input voltage",
            "input-voltage": "input voltage",
            "  INPUT__VOLTAGE  ": "input voltage",
            "input\t  voltage\n": "input voltage",
            "input\u00a0voltage": "input voltage",
            "input\u2011voltage": "input voltage",
            "HTTPStatusCode": "http status code",
            "sensor2Voltage": "sensor2 voltage",
            "temperature_°C": "temperature °c",
            "cafe\u0301_position": "café position",
            "": "",
            "__ -- ": "",
        }
        for original, expected in cases.items():
            with self.subTest(original=original):
                result = normalize_column(original)
                self.assertEqual(result, expected)
                self.assertEqual(normalize_column(result), result)

    def test_semantic_distinctions_survive(self):
        names = ["input_voltage", "incoming_voltage", "output_voltage",
                 "wheel_position", "tyre_position", "temp", "temperature",
                 "volatge", "voltage", "sensor1", "sensor2"]
        self.assertEqual(len({normalize_column(n) for n in names}), len(names))

    def test_non_string_is_rejected(self):
        with self.assertRaises(TypeError):
            normalize_column(None)


if __name__ == "__main__":
    unittest.main()
