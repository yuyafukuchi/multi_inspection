import json
import numpy as np
from contextlib import redirect_stdout
from os import devnull, listdir
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
from multi_comparison import MultiComparison
from multi_inspection_result import MultiInspectionResult
from novelty_detector import NoveltyDetector
from project import Dataset, Project


class NotTrainedError(Exception):
    pass


class MultiInspector:
    
    def __init__(self, project: Project, target_area_tags: list):
        self.project = project
        self.target_area_tags = target_area_tags
        self.multi_comparison = MultiComparison()
        self.train_ok_dist_dict = {}

    # ---------- Train ---------- #

    def multi_train(self, n_jobs: int = -1, predict_scores: bool = True):  # n_jobs=-1 で最大の並列処理数
        model_dict = self.__load_model(load_weights=False)
        Parallel(n_jobs=n_jobs)([delayed(self.__train)(str(target_area_tag), model_dict[target_area_tag], predict_scores)
                                 for target_area_tag
                                 in self.target_area_tags])

    def __train(self, target_area_tag: str, model: NoveltyDetector, predict_scores: bool):
        print('### training start: ' + target_area_tag + ' ###')
        # with redirect_stdout(open(devnull, 'w')):  # 出力を消す
        if True:
            # FIXME: Be compatible with images other than jpg (png, etc.)
            model.fit_paths(list(self.project.dataset.image_directory(Dataset.Category.TRAIN_OK, target_area_tag).glob('*.jpg')))   # image_directoryの引数directory_nameで使用フォルダ指定可能(default: preprocessed)
            model.save(self.project.model_path(area_tag=target_area_tag))
        print('weight saved: '+target_area_tag)
        if predict_scores:
            self.train_ok_dist_dict[target_area_tag] = self.__predict(target_area_tag, model, Dataset.Category.TRAIN_OK)
        print('train finished: '+target_area_tag)

    # ---------- Test ---------- #

    def multi_test(self, n_jobs: int = -1):  # n_jobs=-1 で最大の並列処理数
        model_dict = self.__load_model(load_weights=True)
        try:
            self.load_train_ok_dists()
        except NotTrainedError:
            for target_area_tag in self.target_area_tags:
                self.train_ok_dist_dict[target_area_tag] = self.__predict(target_area_tag, model_dict[target_area_tag], Dataset.Category.TRAIN_OK)

        Parallel(n_jobs=n_jobs)([delayed(self.__test)(str(target_area_tag), model_dict[target_area_tag])
                                 for target_area_tag
                                 in self.target_area_tags])

    def __test(self, target_area_tag: str, model: NoveltyDetector):
        print('### test start: ' + str(target_area_tag) + ' ###')
        with redirect_stdout(open(devnull, 'w')):  # 出力を消す
            test_ok_dists = self.__predict(target_area_tag, model, Dataset.Category.TEST_OK)
            try:
                test_ng_dists = self.__predict(target_area_tag, model, Dataset.Category.TEST_NG, make_directory=False)
            except:
                test_ng_dists = {}

            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.figure()
            sns.distplot(self.train_ok_dist_dict[target_area_tag]['scores'], kde=False, rug=False, label='TRAIN OK', color='g')
            sns.distplot(test_ok_dists['scores'], kde=False, rug=False, label='TEST OK', color='b')
            if test_ng_dists:
                sns.distplot(test_ng_dists['scores'], kde=False, rug=False, label='TEST NG', color='r')
            plt.title('Novelty detection on {}th layer on {} and {} in area {}'.format(
                18, 'vgg16', 'svm', target_area_tag)
            )
            plt.xlabel('Signed distance to hyper plane')
            plt.ylabel('a.u.')
            plt.legend()
            plt.savefig(self.project.histogram_path(area_tag=target_area_tag))
        print('test finished: '+target_area_tag)

    def __predict(self, target_area_tag: str, model: NoveltyDetector, category: Dataset.Category, make_directory: bool = True) -> dict:
        # FIXME: Be compatible with images other than jpg (png, etc.)
        scores = model.predict_paths(list(self.project.dataset.image_directory(category, target_area_tag).glob('*.jpg')))   # image_directoryの引数directory_nameで使用フォルダ指定可能(default: preprocessed)
        distances = {
            'scores': list(scores),
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores))  # standard deviation
        }
        distances_json_path = self.project.distances_json_path(area_tag=target_area_tag,
                                                               category=category,
                                                               make_directory=make_directory)
        with distances_json_path.open(mode='w') as fw:
            json.dump(distances, fw, indent=4)
        return distances

    # ---------- Inspect ---------- #

    def multi_inspect(self, capture_id: str, method_name: str = 'fdr_bh', alpha: float = 0.0013) -> MultiInspectionResult:
        model_dict = self.__load_model(load_weights=True)
        distances, p_values = self.__inspect(capture_id=capture_id, model_dict=model_dict)
        inspection_result = self.multi_comparison.do_multi_test(method_name=method_name,
                                                     area_tags=self.target_area_tags,
                                                     distances=distances,
                                                     p_values=p_values,
                                                     alpha=alpha)
        inspection_result.save(self.project.inspection_result_json_path(capture_id=capture_id))
        return inspection_result

    def __inspect(self, capture_id: str, model_dict: dict) -> Tuple[Dict, List]:
        distances = {}
        p_values = []
        for target_area_tag in self.target_area_tags:
            image_path = self.project.inspection_image_path(capture_id=capture_id, area_tag=target_area_tag)
            distances[target_area_tag] = model_dict[target_area_tag].predict_paths([str(image_path)])[0]
            p_values.append(self.multi_comparison.calc_p_value(data_dist=distances[target_area_tag],
                                                               mean=self.train_ok_dist_dict[target_area_tag]['mean'],
                                                               std=self.train_ok_dist_dict[target_area_tag]['std']))
        return distances, p_values

    def __load_model(self, load_weights: bool) -> dict:
        model_dict = {}
        for target_area_tag in self.target_area_tags:
            detector = NoveltyDetector()
            if load_weights:
                detector.load(str(self.project.model_path(area_tag=target_area_tag)))
            model_dict[target_area_tag] = detector
        return model_dict

    def load_train_ok_dists(self):
        for target_area_tag in self.target_area_tags:
            if target_area_tag in self.train_ok_dist_dict:
                continue
            distances_json_path = self.project.distances_json_path(area_tag=target_area_tag,
                                                                   category=Dataset.Category.TRAIN_OK)
            if distances_json_path.exists():
                with distances_json_path.open(mode='r') as f:
                    self.train_ok_dist_dict[target_area_tag] = json.load(f)
            else:
                raise NotTrainedError("This MultiInspector instance is not trained yet. Call 'multi_train' with appropriate arguments before using this method.")


if __name__ == '__main__':
    from pathlib import Path
    project_root_path = input("Enter the project root path:")
    project = Project(Path(project_root_path))
    # project.set_up()
    # TODO: delete sorting area tags
    inspector = MultiInspector(project=project, target_area_tags=sorted(project.all_area_tags(directory_name='preprocessed')))

    if input('Do you train? (y or n):')=='y':
        inspector.multi_train(n_jobs=-1)
    if input('Do you test? (y or n):')=='y':
        inspector.multi_test(n_jobs=-1)

    inspector.load_train_ok_dists()
    
    print('Inspection starts.')
    alpha = input('alpha value (default: 0.0013):')
    try:
        alpha = float(alpha)
    except:
        alpha = 0.013
    while True:
        capture_id = input('capture id (q:quit):')
        if capture_id == 'q':
            break 
        elif not capture_id in listdir(project_root_path+'/inspection_targets'):
            print('This capture id is not in the inspection_targets directory.')
            continue
        inspection_result = inspector.multi_inspect(capture_id=str(capture_id), alpha=alpha)
        print(inspection_result.is_positive())
