package practicas.po3;

import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.EuclideanDistance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.neighboursearch.LinearNNSearch;

import static utils.Utils.*;
import static weka.classifiers.lazy.IBk.TAGS_WEIGHTING;
import static weka.classifiers.lazy.IBk.WEIGHT_NONE;

public class PO3Main {

    /**
     * args[0] = data path
     * args[1] = class index
     * args[2] = results.txt path
     * args[3] = cls.model path
     * @param args
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 4) {
            System.err.println("4 args are needed.");
            System.exit(1);
        }

        String dataPath = args[0];
        int classIndex = Integer.parseInt(args[1]);
        String resultsPath = args[2];
        String modelPath = args[3];

        Instances instances = loadInstances(dataPath);
        instances.setClassIndex(classIndex);

        IBk cls;

        // Evaluar usando los parametros del barrido
        cls = getBestIBk();
        Evaluation eval = crossvalidation(cls, instances, 10);
        String results = getEvaluationResults(eval);
        printToFile(results, resultsPath);

        // Construir y guardar el modelo entrenado con todos los datos
        cls = getBestIBk();
        cls.buildClassifier(instances);
        saveModel(cls, modelPath);

        // Cargar un modelo ya guardado
        IBk clsCargado = (IBk) loadModel(modelPath);
        Instances dev = split(10.0, instances, false); // En realidad deberia usar un arff de "dev"
                                                                      // pero es para probar que va.
        Evaluation anotherEval = new Evaluation(instances);
        // double onePrediction = anotherEval.evaluateModelOnce(cls, oneInstance); // evaluar una instancia
        double[] predictions = anotherEval.evaluateModel(clsCargado, dev);  // evaluar un conjunto
        String predResults = "";
        for (double d: predictions) {
            System.out.println(d);
            predResults += d;
            predResults += "\n";
        }
        System.out.println(predResults);  // puede que pida guardarlo en un archivo
    }

    /**
     * Para hacer esta funcion ya sabemos los parametros adecuados gracias al barrido.
     * @return
     * @throws Exception
     */
    public static IBk getBestIBk() throws Exception {
        IBk cls = new IBk(8);
        LinearNNSearch search = new LinearNNSearch();
        search.setDistanceFunction(new EuclideanDistance());
        cls.setNearestNeighbourSearchAlgorithm(search);
        cls.setDistanceWeighting(new SelectedTag(WEIGHT_NONE, TAGS_WEIGHTING));
        return cls;
    }

}
