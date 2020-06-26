const path = require("path");

module.exports = {
  mode: "development",
  entry: path.resolve(__dirname, "src") + "/index.js",
  devtool: "inline-source-map",
  devServer: {
    port: 8081,
    contentBase: "./dist",
  },
  output: {
    filename: "[name].bundle.js",
    path: path.resolve(__dirname, "dist"),
    publicPath: "/",
  },
  module: {
    rules: [
      {
        test: /opencv.js$/i,
        use: [
          {
            loader: "raw-loader",
            options: {
              esModule: false,
            },
          },
        ],
      },
    ],
  },
};
