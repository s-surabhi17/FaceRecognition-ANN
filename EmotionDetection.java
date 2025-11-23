import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

public class EmotionDetection {

    static final int NUM_CLASSES = 6;
    static final int INPUT_SIZE = 48 * 48;
    static final int HIDDEN_UNITS = 64;

    public static void main(String[] args) {

        String trainPath = "C:\\visualize\\trainoutput2.txt";
        String testPath = "C:\\visualize\\6emotionsacuuracy.txt";

        int numTrain = 500;
        int numTest = 15;

        double[][] trainX = new double[numTrain][INPUT_SIZE];
        int[] trainY = new int[numTrain];

        double[][] testX = new double[numTest][INPUT_SIZE];
        int[] testY = new int[numTest];

        loadDataset(trainPath, trainX, trainY, numTrain);
        loadDataset(testPath, testX, testY, numTest);

        double[][] W1 = new double[HIDDEN_UNITS][INPUT_SIZE + 1];
        double[][] W2 = new double[NUM_CLASSES][HIDDEN_UNITS + 1];

        initialize(W1);
        initialize(W2);

        train(trainX, trainY, W1, W2, 0.01, 200);

        double acc = test(testX, testY, W1, W2);
        System.out.println("Final Accuracy = " + (acc * 100) + "%");
    }

    private static void loadDataset(String path, double[][] X, int[] y, int rows) {
        try {
            Scanner sc = new Scanner(new File(path));
            for (int i = 0; i < rows; i++) {
                String[] p = sc.nextLine().split(" ");
                y[i] = Integer.parseInt(p[0]);
                for (int j = 0; j < INPUT_SIZE; j++) {
                    X[i][j] = Double.parseDouble(p[j + 1]) / 255.0;
                }
            }
            sc.close();
        } catch (FileNotFoundException e) {
            System.out.println("Dataset missing!");
        }
    }

    private static void initialize(double[][] W) {
        for (int i = 0; i < W.length; i++)
            for (int j = 0; j < W[0].length; j++)
                W[i][j] = (Math.random() - 0.5) * 0.1;
    }

    private static double[] sigmoidLayer(double[] inp, double[][] W) {
        double[] out = new double[W.length];
        for (int i = 0; i < W.length; i++) {
            double s = W[i][inp.length];
            for (int j = 0; j < inp.length; j++) s += W[i][j] * inp[j];
            out[i] = 1.0 / (1 + Math.exp(-s));
        }
        return out;
    }

    private static double[] softmax(double[] z) {
        double max = Arrays.stream(z).max().getAsDouble();
        double sum = 0;
        for (int i = 0; i < z.length; i++) {
            z[i] = Math.exp(z[i] - max);
            sum += z[i];
        }
        for (int i = 0; i < z.length; i++) z[i] /= sum;
        return z;
    }

    private static double[] outputLayer(double[] inp, double[][] W) {
        double[] out = new double[W.length];
        for (int i = 0; i < W.length; i++) {
            double s = W[i][inp.length];
            for (int j = 0; j < inp.length; j++) s += W[i][j] * inp[j];
            out[i] = s;
        }
        return softmax(out);
    }

    private static void train(double[][] X, int[] Y, double[][] W1, double[][] W2, double lr, int epochs) {

        for (int ep = 1; ep <= epochs; ep++) {
            double loss = 0;

            for (int i = 0; i < X.length; i++) {

                double[] H = sigmoidLayer(X[i], W1);
                double[] O = outputLayer(H, W2);

                double[] target = new double[NUM_CLASSES];
                target[Y[i]] = 1.0;

                double[] dO = new double[NUM_CLASSES];
                for (int c = 0; c < NUM_CLASSES; c++) {
                    dO[c] = target[c] - O[c];
                    loss += -target[c] * Math.log(O[c] + 1e-10);
                    for (int h = 0; h < H.length; h++) W2[c][h] += lr * dO[c] * H[h];
                    W2[c][H.length] += lr * dO[c];
                }

                double[] dH = new double[HIDDEN_UNITS];
                for (int h = 0; h < HIDDEN_UNITS; h++) {
                    double sum = 0;
                    for (int c = 0; c < NUM_CLASSES; c++) sum += W2[c][h] * dO[c];
                    dH[h] = H[h] * (1 - H[h]) * sum;
                    for (int p = 0; p < INPUT_SIZE; p++) W1[h][p] += lr * dH[h] * X[i][p];
                    W1[h][INPUT_SIZE] += lr * dH[h];
                }
            }

            if (ep % 10 == 0)
                System.out.println("Epoch " + ep + " loss = " + loss);
        }
    }

    private static double test(double[][] X, int[] Y, double[][] W1, double[][] W2) {
        int correct = 0;

        for (int i = 0; i < X.length; i++) {
            double[] H = sigmoidLayer(X[i], W1);
            double[] O = outputLayer(H, W2);

            int pred = 0;
            for (int j = 1; j < NUM_CLASSES; j++) if (O[j] > O[pred]) pred = j;

            System.out.println("Predicted=" + pred + " Actual=" + Y[i]);

            if (pred == Y[i]) correct++;
        }

        return correct / (double) X.length;
    }
}
