package utils;

import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.core.Attribute;
import weka.core.Instances;

import weka.core.SerializationHelper;
import weka.core.converters.ConverterUtils.DataSink;

import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.filters.unsupervised.instance.Resample;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.Iterator;
import java.util.Random;

public class Utils {

    /**
     * Carga las instancias.
     * @param path la ruta de donde coger las instancias
     * @return
     */
    public static Instances loadInstances(String path) {
        DataSource source = null;
        try {
            source = new DataSource(path);
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR: el archivo no se ha encontrado");
        }

        Instances instances = null;
        try {
            assert source != null;
            instances = source.getDataSet();
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println("ERROR: no se han podido cargar las instancias");
        }

        return instances;
    }


    /**
     * Devuelve informacion del conjunto de datos.
     * @param instances
     * @return
     */
    public static String getInfo(Instances instances) {
        StringBuilder result = new StringBuilder();
        result.append("##### INSTANCES INFO #####");
        result.append("\n");
        result.append("\n");
        result.append("Size: ").append(instances.size());
        result.append("\n");
        result.append("\n");

        result.append("## ATRIBUTES ");
        result.append(instances.numAttributes());
        result.append(" ##");
        result.append("\n");
        Iterator<Attribute> it = instances.enumerateAttributes().asIterator();
        while (it.hasNext()) {

            Attribute attribute = it.next();
            result.append(attribute.toString());
            result.append("\n");
        }
        if (instances.classIndex() >= 0) {
            result.append("CLASS\n");
            result.append(instances.classAttribute().toString());
        }
        result.append("\n");

        return result.toString();
    }


    /**
     * Devuelve un nuevo dataset filtrando los atributos que no son importantes.
     * @param instances
     * @return
     */
    public static Instances filterAttributes(Instances instances) {
        AttributeSelection filter = new AttributeSelection();
        filter.setEvaluator(new CfsSubsetEval()); // Correlation-based feature selection
        filter.setSearch(new BestFirst());
        try {
            filter.setInputFormat(instances);
        } catch (Exception e) {
            e.printStackTrace();
        }

        Instances newData = null;
        try {
            newData = Filter.useFilter(instances, filter);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return newData;
    }


    /**
     * Hace kfold crossvalidation dado un dataset y un clasificador.
     * @param cls clasificador
     * @param instances instancias
     * @param folds numero de iteraciones
     * @return
     * @throws Exception
     */
    public static Evaluation crossvalidation(Classifier cls, Instances instances, int folds) throws Exception {
        Evaluation evaluation = new Evaluation(instances);
        evaluation.crossValidateModel(cls, instances, folds, new Random(1));
        return evaluation;
    }


    /**
     * Hace una evaluacion hold out.
     * @param cls
     * @param train
     * @param test
     * @return
     * @throws Exception
     */
    public static Evaluation holdOutEval(Classifier cls, Instances train, Instances test) throws Exception {
        cls.buildClassifier(train);
        Evaluation evaluation = new Evaluation(train);
        evaluation.evaluateModel(cls, test);
        return evaluation;
    }


    /**
     * Parte un dataset.
     * @param percent por donde partirlo.
     * @param instances el dataset a partir.
     * @param inverse si percent es 70%, que inverse sea true significa
     *                que coge el 70% si inverse es false cogera el 30%.
     * @return
     * @throws Exception
     */
    public static Instances split(Double percent, Instances instances, Boolean inverse) throws Exception {
        Resample filter = new Resample();
        filter.setInputFormat(instances);
        filter.setInvertSelection(inverse);
        filter.setNoReplacement(true);
        filter.setRandomSeed(1);
        filter.setSampleSizePercent(percent);

        return Filter.useFilter(instances, filter);
    }


    /**
     * Hace una particion train dado un dataset y el porcentaje
     * @param percent por donde hacer la particion
     * @param instances el dataset
     * @return
     * @throws Exception
     */
    public static Instances getTrain(Double percent, Instances instances) throws Exception {
        return split(percent, instances, false);
    }


    /**
     * Hace una particion test dado un dataset y el porcentaje
     * @param percent por donde hacer la particion
     * @param instances el dataset
     * @return
     * @throws Exception
     */
    public static Instances getTest(Double percent, Instances instances) throws Exception {
        return split(percent, instances, true);
    }


    /**
     * El output de weka.
     * @param evaluation
     * @return
     * @throws Exception
     */
    public static String getEvaluationResults(Evaluation evaluation) throws Exception {

        //        double acc=evaluation.pctCorrect();
//        double inc=evaluation.pctIncorrect();
//        double kappa=evaluation.kappa();
//        double mae=evaluation.meanAbsoluteError();
//        double rmse=evaluation.rootMeanSquaredError();
//        double rae=evaluation.relativeAbsoluteError();
//        double rrse=evaluation.rootRelativeSquaredError();
//        double confMatrix[][]= evaluation.confusionMatrix();
//
//        System.out.println("Correctly Classified Instances  " + acc);
//        System.out.println("Incorrectly Classified Instances  " + inc);
//        System.out.println("Kappa statistic  " + kappa);
//        System.out.println("Mean absolute error  " + mae);
//        System.out.println("Root mean squared error  " + rmse);
//        System.out.println("Relative absolute error  " + rae);
//        System.out.println("Root relative squared error  " + rrse);
//        System.out.println("Confusion matrix:");
//        System.out.println(confMatrix.);
        String result = evaluation.toSummaryString() +
                "\n" +
                evaluation.toClassDetailsString() +
                "\n" +
                evaluation.toMatrixString() +
                "\n";
        return result;
    }


    /**
     * Guarda instancias en un archivo
     * @param instances las instancias a guardar
     * @param path la ruta a guardar
     */
    public static void saveInstances(Instances instances, String path) {
        try {
            DataSink.write(path, instances);
        }
        catch (Exception e) {
            System.err.println("Failed to save data to: " + path);
            e.printStackTrace();
        }
    }


    /**
     * Escribe texto en un archivo.
     * @param text
     * @param path
     */
    public static void printToFile(String text, String path) {
        try {
            PrintWriter out = new PrintWriter(path);
            out.println(text);
            out.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }


    /**
     * Guarda el modelo en un archivo.
     * @param cls clasificador a guardar
     * @param path rutado donde guardarlo
     * @throws Exception
     */
    public static void saveModel(Classifier cls, String path) throws Exception {
        SerializationHelper.write(path, cls);
    }


    /**
     * Carga el modelo de un archivo.
     * @param path ruta de donde cargarlo
     * @return
     * @throws Exception
     */
    public static Classifier loadModel(String path) throws Exception {
        return (Classifier) SerializationHelper.read(path);
    }
}
