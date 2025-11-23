import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.ListDataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.util.ModelSerializer;

import java.io.File;
import java.util.*;

public class TrainEmotionModelDL4J {

    static final int NUM_CLASSES = 6;
    static final int WIDTH = 48;
    static final int HEIGHT = 48;
    static final int INPUT_SIZE = WIDTH * HEIGHT;

    public static void main(String[] args) throws Exception {

        String trainPath = "data/fer2013train.txt";
        String testPath = "data/fer2013test.txt";

        DataSet trainData = loadDataset(trainPath);
        DataSet testData = loadDataset(testPath);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .list()
                .layer(new DenseLayer.Builder().nIn(INPUT_SIZE).nOut(256).activation(Activation.RELU).build())
                .layer(new DenseLayer.Builder().nOut(128).activation(Activation.RELU).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation(Activation.SOFTMAX).nOut(NUM_CLASSES).build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10));

        ListDataSetIterator<DataSet> trainIter =
                new ListDataSetIterator<>(trainData.asList(), 32);

        System.out.println("Training Started...");
        for (int epoch = 1; epoch <= 30; epoch++) {
            trainIter.reset();
            model.fit(trainIter);
            System.out.println("Epoch " + epoch + " completed");
        }

        evaluate(model, testData);

        File modelFile = new File("models/emotion_model_dl4j.zip");
        ModelSerializer.writeModel(model, modelFile, true);
        System.out.println("Model saved successfully!");
    }

    private static DataSet loadDataset(String path) throws Exception {
        Scanner sc = new Scanner(new File(path));

        ArrayList<double[]> features = new ArrayList<>();
        ArrayList<double[]> labels = new ArrayList<>();

        while (sc.hasNextLine()) {
            String[] parts = sc.nextLine().trim().split(" ");
            int label = Integer.parseInt(parts[0]);

            double[] target = new double[NUM_CLASSES];
            target[label] = 1.0;
            labels.add(target);

            double[] pixels = new double[INPUT_SIZE];
            for (int i = 0; i < INPUT_SIZE; i++)
                pixels[i] = Double.parseDouble(parts[i + 1]) / 255.0;

            features.add(pixels);
        }

        sc.close();
        return new DataSet(Nd4j.create(features.toArray(new double[0][])),
                Nd4j.create(labels.toArray(new double[0][])));
    }

    private static void evaluate(MultiLayerNetwork model, DataSet test) {
        INDArray output = model.output(test.getFeatures());
        INDArray labels = test.getLabels();

        int correct = 0;
        int total = (int) labels.size(0);

        for (int i = 0; i < total; i++) {
            int actual = labels.getRow(i).argMax(1).getInt(0);
            int predicted = output.getRow(i).argMax(1).getInt(0);
            if (actual == predicted) correct++;
        }

        System.out.println("Accuracy = " + (100.0 * correct / total) + "%");
    }
}
