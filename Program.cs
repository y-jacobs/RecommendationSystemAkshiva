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
            TrainModel(args[0], args[1]);
        }   
    }
}