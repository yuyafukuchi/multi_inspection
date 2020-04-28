from enum import auto, Enum
from pathlib import Path
from typing import Callable, List, Tuple
from numpy import ndarray


class Dataset:

    class Category(Enum):
        TRAIN_OK = auto()
        TEST_OK = auto()
        TEST_NG = auto()

        @classmethod
        def all(cls) -> Tuple['Dataset.Category', 'Dataset.Category', 'Dataset.Category']:
            return cls.TRAIN_OK, cls.TEST_OK, cls.TEST_NG

    def __init__(self, root_path: Path):
        self.root_path = root_path

    def set_up(self):
        self.__dataset_path(directory_name='original_plain').mkdir(parents=True)
        self.__dataset_path(directory_name='original_printed').mkdir(parents=True)
        self.__dataset_path(directory_name='preprocessed').mkdir(parents=True)

    def all_area_tags(self, directory_name: str) -> List[str]:
        dataset_path = self.__dataset_path(directory_name=directory_name)
        return [area_path.name for area_path in dataset_path.iterdir()
                if area_path.is_dir() and not area_path.name.startswith('.')]

    @classmethod
    def image_name(cls, capture_id: str, area_tag: str) -> str:
        return f"{capture_id}_{area_tag}.jpg"

    def image_directory(self,
                        category: 'Dataset.Category',
                        area_tag: str,
                        directory_name: str = 'preprocessed',
                        make_directory: bool = False) -> Path:
        area_path = self.__dataset_path(directory_name=directory_name) / area_tag
        if category == Dataset.Category.TRAIN_OK:
            directory = area_path / "train" / "OK"
        elif category == Dataset.Category.TEST_OK:
            directory = area_path / "test" / "OK"
        elif category == Dataset.Category.TEST_NG:
            directory = area_path / "test" / "NG"
        else:
            assert False

        if make_directory and not directory.exists():
            directory.mkdir(parents=True)
        return directory

    def image_path(self,
                   category: 'Dataset.Category',
                   capture_id: str,
                   area_tag: str,
                   directory_name: str = 'preprocessed',
                   make_directory: bool = False) -> Path:
        image_directory = self.image_directory(category=category,
                                               area_tag=area_tag,
                                               directory_name=directory_name,
                                               make_directory=make_directory)
        return image_directory / Dataset.image_name(capture_id=capture_id, area_tag=area_tag)

    def __dataset_path(self, directory_name: str):
        return self.root_path / directory_name


class DatasetProcessor:

    def __init__(self, dataset: Dataset, image_processor: Callable[[ndarray, str], ndarray]):
        """

        :param dataset: Processing dataset
        :param image_processor: Image processing function.
                                It receives an original image(ndarray) and area tag(str), and returns the processed image.
        """

        self.dataset = dataset
        self.image_processor = image_processor

    # def preprocess(self):
    #     print("Start dataset preprocessing...")
    #     from joblib import Parallel, delayed
    #     for category in Dataset.Category.all():
    #         Parallel(n_jobs=-1)([delayed(self.__preprocess)(area_tag, category)
    #                              for area_tag in self.dataset.all_area_tags(directory_name=False)])
    #     print("End dataset preprocessing.")

    # def __preprocess(self, area_tag: str, category: Dataset.Category):
    #     import cv2
    #     image_directory = self.dataset.image_directory(category, area_tag, directory_name=False)
    #     # FIXME: Be compatible with images other than jpg (png, etc.)
    #     for original_image_path in image_directory.glob(pattern='*.jpg'):
    #         original_image = cv2.imread(filename=str(original_image_path))
    #         original_image_name = original_image_path.name
    #         processed_image = self.image_processor(original_image, area_tag)
    #         processed_image_path = self.dataset.image_directory(category=category,
    #                                                             area_tag=area_tag,
    #                                                             directory_name=True,
    #                                                             make_directory=True) / original_image_name
    #         cv2.imwrite(filename=str(processed_image_path), img=processed_image)


class Project:

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.dataset = Dataset(root_path=self.root_path / "dataset")

    # ---------- Setup ---------- #

    def set_up(self):
        self.__models_path.mkdir(parents=True)
        self.__distances_path.mkdir(parents=True)
        self.__histograms_path.mkdir(parents=True)
        self.__inspection_targets_path.mkdir(parents=True)
        self.dataset.set_up()

    def all_area_tags(self, directory_name: str) -> List[str]:
        return self.dataset.all_area_tags(directory_name=directory_name)

    # ---------- Model ---------- #

    def model_path(self, area_tag: str) -> Path:
        return self.__models_path / f"sample_{area_tag}.joblib"

    @property
    def __models_path(self) -> Path:
        return self.root_path / "models"

    # ---------- Distribution ---------- #

    def __distances_json_for_area_path(self, area_tag: str) -> Path:
        return self.__distances_path / area_tag

    def distances_json_path(self,
                            category: 'Dataset.Category',
                            area_tag: str,
                            make_directory: bool = False) -> Path:
        area_path = self.__distances_json_for_area_path(area_tag=area_tag)
        if make_directory and not area_path.exists():
            area_path.mkdir(parents=True)
        if category == Dataset.Category.TRAIN_OK:
            return area_path / "train_ok.json"
        elif category == Dataset.Category.TEST_OK:
            return area_path / "test_ok.json"
        elif category == Dataset.Category.TEST_NG:
            return area_path / "test_ng.json"
        else:
            assert False

    def histogram_path(self, area_tag: str) -> Path:
        return self.__histograms_path / f"histogram_{area_tag}.png"

    @property
    def __distributions_path(self) -> Path:
        return self.root_path / "distributions"

    @property
    def __distances_path(self) -> Path:
        return self.__distributions_path / "distances"

    @property
    def __histograms_path(self) -> Path:
        return self.__distributions_path / "histograms"

    # ---------- Inspection ---------- #

    @classmethod
    def inspection_image_name(cls, capture_id: str, area_tag: str) -> str:
        return Dataset.image_name(capture_id=capture_id, area_tag=area_tag)  # Same as dataset image name

    def inspection_images_directory(self, capture_id: str) -> Path:
        return self.__inspection_targets_path / capture_id

    def inspection_image_path(self, capture_id: str, area_tag: str, make_directory: bool = False) -> Path:
        images_path = self.inspection_images_directory(capture_id=capture_id)
        if make_directory and not images_path.exists():
            images_path.mkdir(parents=True)
        return images_path / Project.inspection_image_name(capture_id=capture_id, area_tag=area_tag)

    def inspection_result_json_path(self, capture_id: str) -> Path:
        return self.inspection_images_directory(capture_id=capture_id) / "result.json"

    @property
    def __inspection_targets_path(self) -> Path:
        return self.root_path / "inspection_targets"


if __name__ == '__main__':
    project_root_path = Path(input('Enter a path for the new project: '))
    new_project = Project(root_path=project_root_path)
    new_project.set_up()
    print(f"New project has been created.")
