import json
import sys
from collections import namedtuple
from pathlib import Path
from typing import Dict


class MultiInspectionResult:

    ForArea = namedtuple('ForArea', ('is_positive', 'distance', 'p_value', 'corrected_p_value'))

    def __init__(self, results_for_each_area: Dict[str, ForArea]):
        """
        :param results_for_each_area: Detailed results for each area. Key represents the area tag.
        """
        self.results_for_each_area = results_for_each_area

    def __str__(self):
        description = f"MultiInspectionResult: {'POSITIVE' if self.is_positive() else 'NEGATIVE'}"
        for tag, area in self.results_for_each_area.items():
            description += f"\n - Area \'{tag}\': {'Positive' if area.is_positive else 'Negative'}, distance={area.distance}, p={area.p_value}, corrected_p={area.corrected_p_value}"
        return description

    def is_positive(self) -> bool:
        return any([result.is_positive for result in self.results_for_each_area.values()])

    def save(self, save_location: Path):
        result_json = {
            "is_positive": self.is_positive(),
            "results_for_area": {area_tag: results._asdict()
                                 for area_tag, results in self.results_for_each_area.items()}
        }
        with save_location.open(mode='w') as result_json_file:
            json.dump(result_json, result_json_file)


# for pickling (when the server responds)
setattr(sys.modules[__name__], 'ForArea', MultiInspectionResult.ForArea)
