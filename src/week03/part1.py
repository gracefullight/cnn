from __future__ import annotations

import gzip
import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import feature
from sklearn import metrics, svm

if TYPE_CHECKING:
    from numpy.typing import NDArray


def log(message: str) -> None:
    print(f"[part1] {message}")  # noqa: T201


def should_log_progress(index: int, total: int) -> bool:
    if total <= 0:
        return False

    current = index + 1
    if current == total:
        return True

    percent = (current * 100) // total
    previous_percent = (index * 100) // total
    return percent // 10 > previous_percent // 10


class LocalBinaryPatterns:
    def __init__(self, num_points: int, radius: int) -> None:
        self.num_points = num_points
        self.radius = radius
        self._extract_logged = False

    def extract(self, image: NDArray[np.uint8], eps: float = 1e-7) -> NDArray[np.float64]:
        if not self._extract_logged:
            log("LocalBinaryPatterns.extract() first call")
            self._extract_logged = True

        lbp = feature.local_binary_pattern(
            image,
            self.num_points,
            self.radius,
            method="uniform",
        )  # type: ignore[no-untyped-call]

        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, self.num_points + 3),
            range=(0, self.num_points + 2),
        )

        hist = hist.astype("float")
        hist /= hist.sum() + eps
        return hist


def load_mnist(path: Path, kind: str = "train") -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    log(f"load_mnist(kind='{kind}')")

    labels_path = path / f"{kind}-labels-idx1-ubyte.gz"
    images_path = path / f"{kind}-images-idx3-ubyte.gz"

    with gzip.open(labels_path, "rb") as labels_file:
        labels = np.frombuffer(labels_file.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as images_file:
        images = np.frombuffer(images_file.read(), dtype=np.uint8, offset=16).reshape(
            len(labels),
            784,
        )

    log(f"loaded {len(labels)} samples for kind='{kind}'")
    return images, labels


def main() -> None:  # noqa: PLR0915
    log("main() start")

    # Step 1) 데이터셋 경로 준비
    data_dir = Path(__file__).resolve().parent / "fashion"
    if not data_dir.exists():
        msg = f"Fashion-MNIST 경로를 찾을 수 없습니다: {data_dir}"
        raise FileNotFoundError(msg)

    # Step 2) Fashion-MNIST 데이터 로드 (train / test)
    log("Step 2: load train/test dataset")
    x_train, y_train = load_mnist(data_dir, kind="train")
    x_test, y_test = load_mnist(data_dir, kind="t10k")

    # Step 3) 784 벡터를 28x28 이미지로 복원
    log("Step 3: reshape vectors to 28x28")
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)

    # Step 4) LBP 추출기 설정
    log("Step 4: initialize LBP descriptor")
    descriptor = LocalBinaryPatterns(num_points=24, radius=8)

    # Step 5) 학습 데이터 전체에서 LBP 특징 추출
    log("Step 5: extract LBP features for training set")
    train_features: list[NDArray[np.float64]] = []
    train_labels: list[int] = []
    train_total = len(x_train)
    for idx, (image, label) in enumerate(zip(x_train, y_train, strict=False)):
        hist = descriptor.extract(image)
        train_features.append(hist)
        train_labels.append(int(label))

        if should_log_progress(idx, train_total):
            percent = ((idx + 1) * 100) // train_total
            log(f"Step 5 progress: {min(percent, 100)}% ({idx + 1}/{train_total})")

    x_train_lbp = np.asarray(train_features)
    y_train_lbp = np.asarray(train_labels)

    # Step 6) SVM(RBF, C=100, random_state=42) 학습
    log("Step 6: train SVM classifier")
    model = svm.SVC(kernel="rbf", C=100, random_state=42)
    model.fit(x_train_lbp, y_train_lbp)

    # Step 7) 학습 정확도 확인
    train_acc = model.score(x_train_lbp, y_train_lbp)
    log(f"Train set accuracy: {train_acc:.4f}")

    # Step 8) 테스트 데이터에서 LBP 특징 추출 후 예측
    log("Step 8: extract LBP features for test set and predict")
    test_features: list[NDArray[np.float64]] = []
    test_total = len(x_test)
    for idx, image in enumerate(x_test):
        hist = descriptor.extract(image)
        test_features.append(hist)

        if should_log_progress(idx, test_total):
            percent = ((idx + 1) * 100) // test_total
            log(f"Step 8 progress: {min(percent, 100)}% ({idx + 1}/{test_total})")

    x_test_lbp = np.asarray(test_features)
    predictions = model.predict(x_test_lbp)

    # Step 9) 테스트 정확도 및 혼동행렬 계산
    test_acc = metrics.accuracy_score(y_test, predictions)
    confusion = metrics.confusion_matrix(y_test, predictions)

    log(f"Test set accuracy: {test_acc:.4f}")
    log("Confusion matrix:")
    log(confusion)

    # Step 10) 혼동행렬 시각화 이미지를 파일로 저장
    log("Step 10: save confusion matrix image")
    plt.figure(figsize=(9, 9))
    plt.imshow(confusion, cmap="Blues")
    plt.colorbar()
    for row_index in range(confusion.shape[0]):
        for col_index in range(confusion.shape[1]):
            plt.text(
                col_index,
                row_index,
                f"{confusion[row_index, col_index]:d}",
                ha="center",
                va="center",
            )

    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.title(f"Accuracy Score: {test_acc:.4f}")

    output_path = Path(__file__).resolve().parent / "part1_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Step 11) OpenCV 동작 확인용 샘플 처리
    log("Step 11: run OpenCV sample processing")
    sample_image = x_test[0]
    sample_rgb = cv2.merge([sample_image] * 3)
    sample_rgb = cv2.resize(sample_rgb, (96, 96), interpolation=cv2.INTER_LINEAR)
    _ = sample_rgb.shape

    log(f"Saved confusion matrix image: {output_path}")
    log("main() done")


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    main()
