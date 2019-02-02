package practicas.po1;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

import java.util.Random;

import static utils.Utils.*;

public class PO1Main {

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.out.println("El programa necesita 1 argumento.");
            System.exit(1);
        }

        String dataPath = args[0];

        // 1) Get data
        Instances instances = loadInstances(dataPath);
        instances.randomize(new Random(1));
        instances.setClassIndex(instances.numAttributes() - 1);
        System.out.println(getInfo(instances));

        // 2) Filter attributes
        Instances newData = filterAttributes(instances);
        System.out.println("\n\nDESPUES DE FILTRAR\n");
//        System.out.println(getInfo(newData));


        // 3) Choose Classifier
        NaiveBayes cls = new NaiveBayes();

        // 4) Evaluation (choose one)

        // - a) hold out
        Instances train = getTrain(70.0, newData);
        Instances test = getTest(70.0, newData);
        Evaluation evaluation = holdOutEval(cls, train, test);

        // - b) crossvalidation
//        Evaluation evaluation = crossvalidation(cls, newData, 10);

        // 4.1) Save results
        String results = getEvaluationResults(evaluation);
        System.out.println(results);
        printToFile(results, "results.txt");

        // 5) Build classifier trained with all data and save it
        cls = new NaiveBayes();
        cls.buildClassifier(newData);  // train with all data
        saveModel(cls, "out.model");

        // EXTRA: si hubiera que cargar un modelo y hacer predicciones:
//        NaiveBayes cls_load = (NaiveBayes) loadModel("out.model");
//        Evaluation eval = new Evaluation(newData);
//        double[] result = eval.evaluateModel(cls_load, test);
//        String text = "";
//        for (double i: result) {
//            text += i;
//            text += "\n";
//        }
//        System.out.println(text);
//        printToFile(results, "predictions.txt");
    }
}
