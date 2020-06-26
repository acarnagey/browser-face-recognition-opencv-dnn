import Utils from "./Utils";

import("./opencv.js").then(async (rawModule) => {
  eval.call(null, rawModule.default);
  cv = await cv;
  //   let mat = new cv.Mat();
  //   console.log(mat.size());
  //   mat.delete();
  main();
});

var netDet = undefined,
  netRecogn = undefined,
  cap,
  frame,
  frameBGR,
  camera;
var persons = {};

//! [Run face detection model]
function detectFaces(img) {
  var blob = cv.blobFromImage(
    img,
    1,
    { width: 192, height: 144 },
    [104, 117, 123, 0],
    false,
    false
  );
  netDet.setInput(blob);
  var out = netDet.forward();

  var faces = [];
  for (var i = 0, n = out.data32F.length; i < n; i += 7) {
    var confidence = out.data32F[i + 2];
    var left = out.data32F[i + 3] * img.cols;
    var top = out.data32F[i + 4] * img.rows;
    var right = out.data32F[i + 5] * img.cols;
    var bottom = out.data32F[i + 6] * img.rows;
    left = Math.min(Math.max(0, left), img.cols - 1);
    right = Math.min(Math.max(0, right), img.cols - 1);
    bottom = Math.min(Math.max(0, bottom), img.rows - 1);
    top = Math.min(Math.max(0, top), img.rows - 1);

    if (confidence > 0.5 && left < right && top < bottom) {
      faces.push({
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
      });
    }
  }
  blob.delete();
  out.delete();
  return faces;
}
//! [Run face detection model]

//! [Get 128 floating points feature vector]
function face2vec(face) {
  var blob = cv.blobFromImage(
    face, // image
    1.0 / 255, // scale factor - scales it down 8 times
    { width: 96, height: 96 }, // size
    [0, 0, 0, 0], // mean
    true, // swap red and blue
    false // crop
  );
  debugger;
  netRecogn.setInput(blob); // Sets the new input value for the network
  var vec = netRecogn.forward(); // Runs forward pass to compute output of layer
  blob.delete();
  return vec;
}
//! [Get 128 floating points feature vector]

//! [Recognize]
function recognize(face) {
  var vec = face2vec(face);
  var bestMatchName = "unknown";
  var bestMatchScore = 0.5; // Actually, the minimum is -1 but we use it as a threshold.
  for (name in persons) {
    var personVec = persons[name];
    var score = vec.dot(personVec);

    if (score > bestMatchScore) {
      bestMatchScore = score;
      bestMatchName = name;
    }
  }
  vec.delete();
  return { bestMatchScore, bestMatchName };
}
//! [Recognize]
function recognize2(face) {
  var vec = face2vec(face);
  var bestMatchName = "unknown";
  var bestMatchScore = 0.5; // Actually, the minimum is -1 but we use it as a threshold.

  var vecA = Array.from(vec.data32F);
  var vecB = [
    -0.011544284410774708,
    0.010874264873564243,
    0.04684402793645859,
    0.01617526449263096,
    0.030110953375697136,
    0.15493322908878326,
    -0.06902411580085754,
    -0.1336817443370819,
    -0.04586732015013695,
    -0.09240750968456268,
    0.05097474902868271,
    0.030470270663499832,
    -0.016416354104876518,
    -0.017124323174357414,
    0.025416698306798935,
    -0.03304049372673035,
    -0.06132659316062927,
    0.05056913569569588,
    0.09201578795909882,
    0.08775737136602402,
    0.11629732698202133,
    0.010776747949421406,
    -0.008119720965623856,
    0.07866285741329193,
    0.00034977536415681243,
    0.14139682054519653,
    -0.10448091477155685,
    -0.1955193728208542,
    0.09833766520023346,
    0.00433101924136281,
    0.10936687886714935,
    0.011017469689249992,
    -0.03596009686589241,
    -0.02653745375573635,
    0.14135386049747467,
    0.16619224846363068,
    -0.014958282001316547,
    0.10056327283382416,
    0.08111227303743362,
    -0.07063350826501846,
    0.03485855087637901,
    0.025492388755083084,
    -0.11615346372127533,
    0.005369941703975201,
    -0.07326701283454895,
    0.016030792146921158,
    0.16366399824619293,
    -0.024851715192198753,
    -0.14686618745326996,
    -0.07589920610189438,
    -0.1742483526468277,
    0.02738151140511036,
    0.07649146765470505,
    0.021266331896185875,
    0.07917096465826035,
    0.05795351415872574,
    0.002886747708544135,
    0.1584518998861313,
    -0.13992618024349213,
    -0.09156973659992218,
    -0.06234637647867203,
    0.0770317018032074,
    0.2663736343383789,
    -0.12917804718017578,
    0.14249463379383087,
    0.05014253407716751,
    0.11014106869697571,
    -0.04444137215614319,
    -0.01871700957417488,
    0.12137134373188019,
    0.016965653747320175,
    0.030349107459187508,
    -0.06225559487938881,
    0.03941401466727257,
    0.04520777612924576,
    -0.10297420620918274,
    -0.1149965152144432,
    0.03365776687860489,
    0.06734173744916916,
    0.0065498328767716885,
    -0.06893526017665863,
    0.19026009738445282,
    -0.03827414661645889,
    -0.08895208686590195,
    0.008318080566823483,
    -0.07500004768371582,
    0.03947378322482109,
    0.05226250737905502,
    -0.04288163781166077,
    0.019571030512452126,
    0.22244682908058167,
    -0.012019358575344086,
    0.05657205358147621,
    0.050386954098939896,
    0.03257228061556816,
    -0.12496496737003326,
    0.027670513838529587,
    0.11603370308876038,
    0.023938169702887535,
    -0.0065747154876589775,
    -0.047253891825675964,
    0.0021745075937360525,
    -0.11150266975164413,
    -0.058291442692279816,
    -0.05612559989094734,
    0.1970965415239334,
    0.11454799026250839,
    -0.07244390994310379,
    0.06698980927467346,
    0.000366554333595559,
    0.006348231807351112,
    0.022439200431108475,
    -0.06789188832044601,
    -0.07205997407436371,
    0.12156474590301514,
    0.030761662870645523,
    -0.09823989868164062,
    -0.2112109363079071,
    0.022647138684988022,
    0.03116152063012123,
    0.04862898960709572,
    -0.049545105546712875,
    0.011632212437689304,
    -0.007073790766298771,
    -0.07560490816831589,
    0.16030189394950867,
    0.11356466263532639,
    0.038095179945230484,
  ];
  var score = 0;
  for (let i = 0; i < vecA.length; i++) {
    score += vecA[i] * vecB[i];
  }
  if (score > bestMatchScore) {
    bestMatchScore = score;
    bestMatchName = name;
  }
  vec.delete();
  return { bestMatchScore, bestMatchName: "test" };
}

async function loadModels(callback) {
  // ref: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy_lowres.prototxt"
  // "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel"
  // "https://raw.githubusercontent.com/pyannote/pyannote-data/master/openface.nn4.small2.v1.t7"
  var proto = "/models/deploy_lowres.prototxt.txt";
  var weights = "/models/res10_300x300_ssd_iter_140000_fp16.caffemodel";
  var recognModel = "/models/openface.nn4.small2.v1.t7";
  await createFileFromUrlv2("face_detector.prototxt", proto);
  document.getElementById("status").innerHTML =
    "Downloading face_detector.caffemodel";
  await createFileFromUrlv2("face_detector.caffemodel", weights);
  document.getElementById("status").innerHTML = "Downloading OpenFace model";
  await createFileFromUrlv2("face_recognition.t7", recognModel);
  document.getElementById("status").innerHTML = "";
  // neural network to use for detecting faces
  netDet = cv.readNetFromCaffe(
    "face_detector.prototxt",
    "face_detector.caffemodel"
  );
  // neural network to use for recognizing faces
  netRecogn = cv.readNetFromTorch("face_recognition.t7");
  callback();
}

async function createFileFromUrlv2(path, url) {
  let response = await fetch(url);
  let buffer = await response.arrayBuffer();
  let data = new Uint8Array(buffer);
  cv.FS_createDataFile("/", path, data, true, false, false);
}

function imageDataFromMat(mat) {
  // convert the mat type to cv.CV_8U
  const img = new cv.Mat();
  const depth = mat.type() % 8;
  const scale =
    depth <= cv.CV_8S ? 1.0 : depth <= cv.CV_32S ? 1.0 / 256.0 : 255.0;
  const shift = depth === cv.CV_8S || depth === cv.CV_16S ? 128.0 : 0.0;
  mat.convertTo(img, cv.CV_8U, scale, shift);

  // convert the img type to cv.CV_8UC4
  switch (img.type()) {
    case cv.CV_8UC1:
      cv.cvtColor(img, img, cv.COLOR_GRAY2RGBA);
      break;
    case cv.CV_8UC3:
      cv.cvtColor(img, img, cv.COLOR_RGB2RGBA);
      break;
    case cv.CV_8UC4:
      break;
    default:
      throw new Error(
        "Bad number of channels (Source image must have 1, 3 or 4 channels)"
      );
  }
  const clampedArray = new ImageData(
    new Uint8ClampedArray(img.data),
    img.cols,
    img.rows
  );
  img.delete();
  return clampedArray;
}

function main() {
  //! [Add a person]
  document.getElementById("addPersonButton").onclick = function () {
    var rects = detectFaces(frameBGR);
    if (rects.length > 0) {
      var face = frameBGR.roi(rects[0]);

      var name = prompt("Say your name:");
      var cell = document.getElementById("targetNames").insertCell(0);
      cell.innerHTML = name;

      persons[name] = face2vec(face).clone();

      var canvas = document.createElement("canvas");
      canvas.setAttribute("width", 96);
      canvas.setAttribute("height", 96);
      var cell = document.getElementById("targetImgs").insertCell(0);
      cell.appendChild(canvas);

      var faceResized = new cv.Mat(canvas.height, canvas.width, cv.CV_8UC3);
      cv.resize(face, faceResized, {
        width: canvas.width,
        height: canvas.height,
      });
      cv.cvtColor(faceResized, faceResized, cv.COLOR_BGR2RGB);
      cv.imshow(canvas, faceResized);
      faceResized.delete();
    }
  };
  //! [Add a person]

  //! [Define frames processing]
  var isRunning = false;
  const FPS = 30; // Target number of frames processed per second.

  function captureFrame() {
    var begin = Date.now();
    // debugger;
    cap.read(frame); // Read a frame from camera
    // converts the mat into a blueish/grey color sheme
    cv.cvtColor(frame, frameBGR, cv.COLOR_RGBA2BGR);

    var faces = detectFaces(frameBGR);
    faces.forEach(function (rect) {
      cv.rectangle(
        frame,
        { x: rect.x, y: rect.y },
        { x: rect.x + rect.width, y: rect.y + rect.height },
        [0, 255, 0, 255]
      );

      var face = frameBGR.roi(rect);

      var { bestMatchScore, bestMatchName } = { ...recognize(face) };
      cv.putText(
        frame,
        `${bestMatchName} ${bestMatchScore}`,
        { x: rect.x, y: rect.y },
        cv.FONT_HERSHEY_SIMPLEX,
        1.0,
        [0, 255, 0, 255]
      );
    });

    cv.imshow(output, frame);

    // Loop this function.
    if (isRunning) {
      var delay = 1000 / FPS - (Date.now() - begin);
      setTimeout(captureFrame, delay);
    }
  }
  //! [Define frames processing]

  document.getElementById("startStopButton").onclick = function toggle() {
    if (isRunning) {
      isRunning = false;
      stopCamera();
      document.getElementById("startStopButton").innerHTML = "Start";
      document.getElementById("addPersonButton").disabled = true;
    } else {
      function run() {
        startCamera();

        isRunning = true;
        captureFrame();
        document.getElementById("startStopButton").innerHTML = "Stop";
        document.getElementById("startStopButton").disabled = false;
        document.getElementById("addPersonButton").disabled = false;
      }
      if (netDet == undefined || netRecogn == undefined) {
        document.getElementById("startStopButton").disabled = true;
        loadModels(run); // Load models and run a pipeline;
      } else {
        run();
      }
    }
  };

  document.getElementById("startStopButton").disabled = false;
}

function startCamera() {
  // Create a camera object.
  var output = document.getElementById("output");
  camera = document.createElement("video");
  camera.setAttribute("width", output.width);
  camera.setAttribute("height", output.height);

  // Get a permission from user to use a camera.
  navigator.mediaDevices
    .getUserMedia({ video: true, audio: false })
    .then(function (stream) {
      camera.srcObject = stream;
      camera.onloadedmetadata = function (e) {
        camera.play();
      };
    });

  //! [Open a camera stream]
  //   debugger;
  cap = new cv.VideoCapture(camera);
  frame = new cv.Mat(camera.height, camera.width, cv.CV_8UC4);
  frameBGR = new cv.Mat(camera.height, camera.width, cv.CV_8UC3);
  //! [Open a camera stream]
}

function stopCamera() {
  const stream = camera.srcObject;
  const tracks = stream.getTracks();

  tracks.forEach(function (track) {
    track.stop();
  });

  // camera.srcObject = null;
}

export default Utils;
