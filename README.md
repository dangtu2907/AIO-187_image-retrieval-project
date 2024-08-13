# Công Cụ So Sánh Độ Tương Đồng Hình Ảnh

Công cụ này cho phép bạn so sánh một hình ảnh truy vấn với các hình ảnh trong tập dữ liệu bằng cách sử dụng các phương pháp đo độ tương đồng khác nhau: Khoảng cách L1 (Chênh lệch tuyệt đối), Khoảng cách L2 (Chênh lệch bình phương trung bình), và Độ tương đồng Cosine. Công cụ giúp tìm ra những hình ảnh tương đồng nhất trong tập dữ liệu với hình ảnh truy vấn.

# Yêu Cầu

Trước khi chạy mã, hãy đảm bảo rằng bạn đã cài đặt tập dữ liệu và các thư viện Python sau:
- numpy
- Pillow
- matplotlib
- os

### Bước 1: Tải và giải nén dữ liệu
Bạn cần tải tập tin dữ liệu và giải nén nó. Dùng lệnh sau để tải tập tin:
```python
!gdown 1msLVo0g0LFmL9-qZ73vq9YEVZwbzOePF
!unzip data
```

### Bước 2: Khởi tạo các thư viện và cấu hình
Import các thư viện cần thiết, như `os`, `numpy`, `PIL.Image`, và `matplotlib` để đọc, xử lý và hiển thị hình ảnh:
```python
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
```
Xác định đường dẫn tới thư mục gốc chứa dữ liệu và tên các lớp:
```python
ROOT = 'data'
CLASS_NAME = sorted(list(os.listdir(f'{ROOT}/train')))
```

### Bước 3: Định nghĩa các hàm xử lý hình ảnh
1. **Đọc hình ảnh và thay đổi kích thước:** Đọc hình ảnh từ một đường dẫn nhất định và thay đổi kích thước theo yêu cầu.
   ```python
   def read_image_from_path(path, size):
       im = Image.open(path).convert('RGB').resize(size)
       return np.array(im)
   ```

2. **Chuyển đổi một thư mục thành các mảng hình ảnh:** Đọc toàn bộ hình ảnh trong một thư mục và chuyển đổi chúng thành các mảng NumPy.
   ```python
   def folder_to_images(folder, size):
       list_dir = [folder + '/' + name for name in os.listdir(folder)]
       images_np = np.zeros(shape=(len(list_dir), *size, 3))
       images_path = []
       for i, path in enumerate(list_dir):
           images_np[i] = read_image_from_path(path, size)
           images_path.append(path)
       images_path = np.array(images_path)
       return images_np, images_path
   ```

3. **Hiển thị kết quả tìm kiếm:** Vẽ và hiển thị hình ảnh truy vấn và các hình ảnh tương tự được tìm thấy.
   ```python
   def plot_results(query_path, ls_path_score, reverse):
       fig = plt.figure(figsize=(15, 9))
       fig.add_subplot(2, 3, 1)
       plt.imshow(read_image_from_path(query_path, size=(448,448)))
       plt.title(f"Query Image: {query_path.split('/')[2]}", fontsize=16)
       plt.axis("off")
       for i, path in enumerate(sorted(ls_path_score, key=lambda x : x[1], reverse=reverse)[:5], 2):
           fig.add_subplot(2, 3, i)
           plt.imshow(read_image_from_path(path[0], size=(448,448)))
           plt.title(f"Top {i-1}: {path[0].split('/')[2]}", fontsize=16)
           plt.axis("off")
       plt.show()
   ```

### Bước 4: Xây dựng các hàm tính điểm tương tự
1. **Tính điểm L1 (Absolute Difference):**
   ```python
   def absolute_difference(query, data):
       axis_batch_size = tuple(range(1, len(data.shape)))
       return np.sum(np.abs(data - query), axis=axis_batch_size)
   ```

   **Tìm kiếm hình ảnh tương tự sử dụng L1 Score:**
   ```python
   def get_l1_score(root_img_path, query_path, size):
       query = read_image_from_path(query_path, size)
       ls_path_score = []
       for folder in os.listdir(root_img_path):
           if folder in CLASS_NAME:
               path = root_img_path + folder
               images_np, images_path = folder_to_images(path, size)
               rates = absolute_difference(query, images_np)
               ls_path_score.extend(list(zip(images_path, rates)))
       return query, ls_path_score
   ```

2. **Tính điểm L2 (Mean Square Difference):**
   ```python
   def mean_square_difference(query, data):
       axis_batch_size = tuple(range(1,len(data.shape)))
       return np.mean((data - query)**2, axis=axis_batch_size)
   ```

   **Tìm kiếm hình ảnh tương tự sử dụng L2 Score:**
   ```python
   def get_l2_score(root_img_path, query_path, size):
       query = read_image_from_path(query_path, size)
       ls_path_score = []
       for folder in os.listdir(root_img_path):
           if folder in CLASS_NAME:
               path = root_img_path + folder
               images_np, images_path = folder_to_images(path, size) 
               rates = mean_square_difference(query, images_np)
               ls_path_score.extend(list(zip(images_path, rates)))
       return query, ls_path_score
   ```

3. **Tính độ tương tự Cosine:**
   ```python
   def cosine_similarity(query, data):
       axis_batch_size = tuple(range(1,len(data.shape)))
       query_norm = np.sqrt(np.sum(query**2))
       data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))
       return np.sum(data * query, axis=axis_batch_size) / (query_norm*data_norm + np.finfo(float).eps)
   ```

   **Tìm kiếm hình ảnh tương tự sử dụng Cosine Similarity:**
   ```python
   def get_cosine_similarity_score(root_img_path, query_path, size):
       query = read_image_from_path(query_path, size)
       ls_path_score = []
       for folder in os.listdir(root_img_path):
           if folder in CLASS_NAME:
               path = root_img_path + folder
               images_np, images_path = folder_to_images(path, size)
               rates = cosine_similarity(query, images_np)
               ls_path_score.extend(list(zip(images_path, rates)))
       return query, ls_path_score
   ```

### Bước 5: Thực hiện tìm kiếm hình ảnh
Bây giờ, bạn có thể sử dụng các hàm trên để tìm kiếm hình ảnh tương tự trong tập dữ liệu dựa trên hình ảnh truy vấn.

1. **Sử dụng L1 Score:**
   ```python
   root_img_path = f"{ROOT}/train/"
   query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
   size = (448, 448)
   query, ls_path_score = get_l1_score(root_img_path, query_path, size)
   plot_results(query_path, ls_path_score, reverse=False)
   ```

2. **Sử dụng L2 Score:**
   ```python
   root_img_path = f"{ROOT}/train/"
   query_path = f"{ROOT}/test/African_crocodile/n01697457_18534.JPEG"
   size = (448, 448)
   query, ls_path_score = get_l2_score(root_img_path, query_path, size)
   plot_results(querry_path, ls_path_score, reverse=False)

3. **Sử dụng Cosine Similarity**:
   ```python
   root_img_path = f"{ROOT}/train/"
   query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
   size = (448, 448)
   query, ls_path_score = get_cosine_similarity_score(root_img_path, query_path, size)
   plot_results(query_path, ls_path_score, reverse=True)

### Bước 6: Kết quả
Mỗi phương pháp sẽ hiển thị hình ảnh truy vấn và các hình ảnh tương đồng nhất từ tập dữ liệu dựa trên phương pháp đo lường được sử dụng.