from scipy import stats
from statsmodels.stats.multitest import multipletests
from multi_inspection_result import MultiInspectionResult


class MultiComparison:

    def calc_p_value(self, data_dist, mean: float, std: float) -> list:
        return stats.norm.cdf(x=data_dist, loc=mean, scale=std)

    def do_multi_test(self,
                      method_name: str,
                      area_tags: list,
                      distances: dict,
                      p_values: list,
                      alpha: float) -> MultiInspectionResult:
        rejected, corrected_p_values = multipletests(pvals=p_values, alpha=alpha, method=method_name)[:2]
        results_for_each_area = {area_tag: MultiInspectionResult.ForArea(is_positive=bool(rejected[index]),
                                                                         distance=distances[area_tag],
                                                                         p_value=p_values[index],
                                                                         corrected_p_value=corrected_p_values[index])
                                 for index, area_tag in enumerate(area_tags)}
        return MultiInspectionResult(results_for_each_area)
