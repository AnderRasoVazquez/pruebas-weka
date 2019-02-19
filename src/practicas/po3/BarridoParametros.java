package practicas.po3;

import weka.core.Instances;

import static utils.Utils.*;

public class BarridoParametros {

    /**
     * args:
     * 0 - data path
     * 1 - class index
     * 2 - results.txt path
     *
     * @param args
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("3 args are needed.");
            System.exit(1);
        }

        String dataPath = args[0];
        int classIndex = Integer.parseInt(args[1]);
        String resultsPath = args[2];

        Instances instances = loadInstances(dataPath);
        instances.setClassIndex(classIndex);

        String result = manualSearchBestParamsKNN(instances);
        System.out.println(result);
        printToFile(result, resultsPath);

    }

}
