using Microsoft.ML;
using Microsoft.ML.Trainers;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Google.Cloud.Storage.V1;
using System;
using System.IO;
using Google.Apis.Auth.OAuth2;
using System.Globalization;

namespace RecommendationsSystem
{
    
    class Program
    {
        static void TrainModel(string trainingDataPath, string savedModelPath)
        {
            MLContext mlContext = new MLContext();

            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<RatingData>(trainingDataPath, hasHeader: true, separatorChar: ',');

            trainingDataView = mlContext.Data.Cache(trainingDataView);

            var estimator = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "userIdFeaturized", inputColumnName: nameof(RatingData.userId))
                                          .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "mediaIdFeaturized", inputColumnName: nameof(RatingData.mediaId))
                                          .Append(mlContext.Transforms.Concatenate("Features", "userIdFeaturized", "mediaIdFeaturized")));

            //IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: "userId")
            //        .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "mediaIdEncoded", inputColumnName: "mediaId"));

            var options = new FieldAwareFactorizationMachineTrainer.Options
            {
                FeatureColumnName = "Features",
                ExtraFeatureColumns = new[] { nameof(RatingData.features) },
                LabelColumnName = "rating",
                NumberOfIterations = 20,
            };

            var trainerEstimator = estimator.Append(mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(options));

            Console.WriteLine("=============== Training the model ===============");

            ITransformer model = trainerEstimator.Fit(trainingDataView);


            //var predictionEngine = mlContext.Model.CreatePredictionEngine<RatingData, RatingPrediction>(model);
            //RatingData testData = new RatingData() { userId = "6", mediaId = "10" };

            //var movieRatingPrediction = predictionEngine.Predict(testData);
            //Console.WriteLine($"UserId:{testData.userId} with movieId: {testData.mediaId} Score:{Sigmoid(movieRatingPrediction.Score)} label:{movieRatingPrediction.PredictedLabel}");
            //Console.WriteLine();

            Console.WriteLine("=============== Saving the model ===============");

            mlContext.Model.Save(model, trainingDataView.Schema, savedModelPath);
        }

        static void Evaluate(MLContext mlContext, IDataView testDataView, ITransformer model)
        {
            Console.WriteLine("=============== Evaluating the model ===============");
            var prediction = model.Transform(testDataView);

            var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

            Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
            Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
        }

        public static float Sigmoid(float x)
        {
            return (float)(100 / (1 + Math.Exp(-x)));
        }


        static void Main(String[] args)
        {
            // Console.WriteLine("testttttt");
            // TrainModel("C:/Users/dossifyapp/Desktop/project/RatingData.csv", "C:/Users/dossifyapp/Desktop/project/recommendation_model.zip");
            //TrainModel("C:/Users/yahel/Desktop/RatingData.csv", "C:/Users/yahel/Desktop/recommendation_model.zip");
            TrainModel(args[0], args[1]);
        }   
    }
}