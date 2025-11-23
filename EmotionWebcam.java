import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;

import org.bytedeco.opencv.opencv_core.*;
import org.bytedeco.opencv.opencv_videoio.VideoCapture;
import org.bytedeco.opencv.global.opencv_imgproc;
import org.bytedeco.opencv.global.opencv_highgui;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.bytedeco.opencv.global.opencv_objdetect.CASCADE_SCALE_IMAGE;

public class EmotionWebcam {

    static final int WIDTH = 48, HEIGHT = 48, INPUT_SIZE = WIDTH * HEIGHT;
    static final String[] EMOTIONS = {"Angry","Disgust","Fear","Happy","Sad","Surprise"};

    public static void main(String[] args) throws Exception {

        MultiLayerNetwork model =
                ModelSerializer.restoreMultiLayerNetwork("models/emotion_model_dl4j.zip");

        CascadeClassifier detector =
                new CascadeClassifier("data/haarcascade_frontalface_default.xml");

        VideoCapture cam = new VideoCapture(0);
        if (!cam.isOpened()) { System.out.println("Camera not found!"); return; }

        Mat frame = new Mat();
        Mat gray = new Mat();

        while (true) {
            cam.read(frame);
            cvtColor(frame, gray, COLOR_BGR2GRAY);

            RectVector faces = new RectVector();
            detector.detectMultiScale(gray, faces, 1.1, 3, CASCADE_SCALE_IMAGE);

            for (int i = 0; i < faces.size(); i++) {
                Rect r = faces.get(i);
                rectangle(frame, r, new Scalar(0,255,0,0));

                Mat face = new Mat(gray, r);
                resize(face, face, new Size(WIDTH, HEIGHT));

                double[] input = new double[INPUT_SIZE];
                int idx = 0;
                for (int y = 0; y < HEIGHT; y++)
                    for (int x = 0; x < WIDTH; x++)
                        input[idx++] = face.ptr(y, x).get() / 255.0;

                INDArray feature = Nd4j.create(input).reshape(1, INPUT_SIZE);
                int pred = model.output(feature).argMax().getInt(0);

                putText(frame, EMOTIONS[pred], new Point(r.x(), r.y() - 10),
                        FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(0,255,0,0));
            }

            opencv_highgui.imshow("Emotion Detector", frame);

            if (opencv_highgui.waitKey(1) == 'q') break;
        }

        cam.release();
        opencv_highgui.destroyAllWindows();
    }
}
