# Book Recommendation System

A collaborative filtering-based book recommendation system built using the Surprise library. The system predicts user preferences and provides personalized book recommendations based on historical rating data.

## Project Overview

This project implements a user-specific recommendation model that predicts which books individual users are likely to appreciate, based on their previous ratings and the behavior of similar users.

## Dataset

### Books Dataset
Books are identified by their ISBN codes. The dataset includes:
- **Book-Title**: Title of the book
- **Book-Author**: Author name (only the first author if multiple)
- **Year-Of-Publication**: Publication year
- **Publisher**: Publisher name
- **Image-URL-S/M/L**: Cover image URLs in three sizes (small, medium, large) linking to Amazon

Data source: Amazon Web Services

### Ratings Dataset
Contains book rating information with two types of ratings:
- **Explicit ratings**: Scale of 1–10 (higher value = better rating)
- **Implicit ratings**: Indicated by value 0 (user has not provided a numerical rating)

### Data Statistics
- Original dataset: 1,031,136 rows × 4 columns
- Invalid ISBN values: 85,392 rows (~8% of data)
- After cleaning and filtering: ~945,744 rows
- Unique books (ISBNs): 74,666
- Rating scale: 1-10

## Project Components

### 1. Data Preprocessing and Quality Checking
- Merging book and rating data
- Removing invalid ISBN codes
- Handling implicit entries (0-ratings)
- Filtering infrequent users and books to optimize memory usage

### 2. Building Recommendation Models

#### Memory Optimization
Initial training attempts resulted in memory errors due to the large dataset size:
```
MemoryError: Unable to allocate 20.8 GiB for an array with shape (74666, 74666) and data type int32
```

**Solution**: Implemented filtering strategy:
- Minimum ratings per user: 5
- Minimum ratings per book: 10

This reduced the matrix dimensions while retaining the most informative data.

#### Models Implemented

**KNN (K-Nearest Neighbors)**
- Algorithm: KNNBasic (item-based)
- Similarity metric: Pearson correlation
- Optimal k: 30 neighbors
- Hyperparameter optimization via GridSearchCV with 3-fold cross-validation

**SVD (Singular Value Decomposition)**
- Matrix factorization approach
- Learns latent factors representing hidden features (genre, style, popularity)
- Hyperparameters: n_factors, n_epochs, lr_all, reg_all

### 3. Model Evaluation

Performance metrics on test set:

| Model | RMSE | MAE |
|-------|------|-----|
| **KNN (optimized)** | **0.852** | **0.557** |
| **SVD** | 1.585 | 1.218 |
| SVD (before optimization) | 3.586 | 2.864 |

#### Key Findings
- **Zero-rating removal** significantly improved model quality
- **KNN model** achieved the best accuracy due to:
  - Pearson similarity accounting for user rating biases
  - Optimal k=30 balancing relevant neighbors
  - Item-based approach leveraging dense data structure (~23 ratings per book)
- **SVD model** provides good generalization and works well for top-N recommendations

### 4. Generating Predictions and Recommendations
- Anti-test set creation for unrated books
- User-specific **Top-N recommendations** (Top-5)
- Single user-book pair prediction capability

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/book-recommendation-system.git
cd book-recommendation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- pandas
- numpy
- scikit-surprise
- jupyter

## Usage

### Running the Jupyter Notebook

```bash
jupyter lab
```

Open the main notebook and run cells sequentially.

### Example: Get Recommendations for a User

```python
# Generate top-5 recommendations
predictions_sample = algo_svd.test(anti_testset_sample)
top_n = get_top_n(predictions_sample, n=5)

# Display recommendations
for user_id in top_n:
    print(f"User: {user_id}")
    print("Recommendations:")
    for isbn, predicted_rating in top_n[user_id]:
        book_title = isbn_to_title.get(isbn, "Unknown book")
        print(f"  • {book_title} (Predicted: {predicted_rating:.1f}/10)")
```

### Example: Predict Single Rating

```python
# Predict rating for specific user-book pair
prediction = algo_svd.predict(user_id, isbn)
print(f"Predicted rating: {prediction.est:.1f}/10")
```

## Results and Analysis

### Performance Improvements
1. **Data cleaning** (zero-rating removal): RMSE improved from 3.586 to 1.585 (SVD)
2. **Hyperparameter optimization**: KNN RMSE improved to 0.852 (40% reduction vs SVD)
3. **Memory optimization**: Successfully trained on standard hardware

### Model Comparison
- **KNN**: Best for accurate rating predictions on dense data
- **SVD**: Better for generalization and sparse datasets
- **Recommendation**: Use KNN for this dataset, SVD for broader applications

## Future Improvements

### Model Enhancements
- **SVD**: Optimize latent factors (n_factors: 20, 50, 100)
- **SVD**: Fine-tune hyperparameters (n_epochs, lr_all, reg_all)
- **KNN**: Experiment with different k values and similarity methods

### Alternative Models
- **SVD++**: Enhanced SVD with implicit feedback
- **BaselineOnly**: Baseline estimators
- **KNNBaseline**: KNN with baseline estimates

### Hybrid Approaches
- Content-based filtering (book metadata)
- Hybrid models combining collaborative and content-based methods
- Deep learning approaches (Neural Collaborative Filtering)

### Author

**Tauimonen**

- GitHub: [@tauimonen](https://github.com/tauimonen)
- Project Link: [https://github.com/tauimonen/book-recommendation-system](https://github.com/tauimonen/ML-book-recommendation)

---

## License

This project is licensed under the MIT License.

## Acknowledgments

- Dataset provided by Amazon Web Services
- Built with [Surprise](http://surpriselib.com/) library
- Inspired by recommender system research and best practices

---

**Note**: This is a prototype-level recommendation system developed for educational purposes. The dataset files are not included in this repository due to size constraints.
