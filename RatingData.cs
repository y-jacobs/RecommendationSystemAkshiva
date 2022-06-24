using Microsoft.ML.Data;

namespace RecommendationsSystem
{
    public class RatingData
    {
        [LoadColumn(0)]
        public string userId;
        [LoadColumn(1)]
        public string mediaId;
        [LoadColumn(2)]
        public bool rating;
        [LoadColumn(3, 11)]
        [VectorType(9)]
        public float[] features;
    }
}
