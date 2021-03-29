const math = require("mathjs");

function sigmoid(x) {
    return (1 / (1 + math.exp(-x)));
}

function deriv_sigmoid(x) {
    return (sigmoid(x) * (1 - sigmoid(x)));
}

function mse_loss(y_true, y_pred, index) {
    var ret = 0;
    var rets = [];
    
    for (var i = 0; i < 4; i++) {
        rets.push(Math.pow(y_true[i] - y_pred[index + i], 2));
    }
    ret = (rets[0] + rets[1] + rets[2] + rets[3]) / 4;
    return(ret);
}

class Neuron {
    constructor(weights, bias) {
        this.weights = weights;
        this.bias = bias;
    }

    feedforward(inputs) {
        return sigmoid(math.dot(this.weights, inputs) + this.bias);
    }
}

class NeuralNetwork {
    constructor () {
        this.weights = [0, 1];
        this.bias = 0;
        this.h1 = new Neuron (this.weights, this.bias);
        this.h2 = new Neuron (this.weights, this.bias);
        this.o1 = new Neuron (this.weights, this.bias);
    }

    feedforward(x) {
        return (this.o1.feedforward([this.h1.feedforward(x), this.h2.feedforward(x)]));
    }
}

class NeuralNetworkLike {
    constructor () {
        this.result = [];
        this.w1 = Math.random() * 2 - 1;
        this.w2 = Math.random() * 2 - 1;
        this.w3 = Math.random() * 2 - 1;
        this.w4 = Math.random() * 2 - 1;
        this.w5 = Math.random() * 2 - 1;
        this.w6 = Math.random() * 2 - 1;
        
        this.b1 = Math.random() * 2 - 1;
        this.b2 = Math.random() * 2 - 1;
        this.b3 = Math.random() * 2 - 1;
    }

    feedforward(x) {
        this.h1 = sigmoid(this.w1 * x[0] + this.w2 * x[1] + this.b1);
        this.h2 = sigmoid(this.w3 * x[0] + this.w4 * x[1] + this.b2);
        return (sigmoid(this.w5 * this.h1 + this.w6 * this.h2 + this.b3));
    }

    train(data, all_y_trues) {
        let learn_rate = 0.1;
        let epochs = 1000;

        for (let epoch = 0; epoch < epochs; epoch++) {
            for (let x, y_true, i = 0; i < 4; i++) {
                x = data[i];
                y_true = all_y_trues[i];
                let sum_h1 = this.w1 * x[0] + this.w2 * x[1] + this.b1;
                this.h1 = sigmoid(sum_h1);

                let sum_h2 = this.w3 * x[0] + this.w4 * x[1] + this.b2;
                this.h2 = sigmoid(sum_h2);

                let sum_o1 = this.w5 * this.h1 + this.w6 * this.h2 + this.b3;
                this.o1 = sigmoid(sum_o1);
                let y_pred = this.o1;

                let d_L_d_ypred = -2 * (y_true - y_pred)

                // Neuron o1
                let d_ypred_d_w5 = this.h1 * deriv_sigmoid(sum_o1)
                let d_ypred_d_w6 = this.h2 * deriv_sigmoid(sum_o1)
                let d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                let d_ypred_d_h1 = this.w5 * deriv_sigmoid(sum_o1)
                let d_ypred_d_h2 = this.w6 * deriv_sigmoid(sum_o1)

                // Neuron h1
                let d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                let d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                let d_h1_d_b1 = deriv_sigmoid(sum_h1)

                // Neuron h2
                let d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                let d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                let d_h2_d_b2 = deriv_sigmoid(sum_h2)

                // --- Update weights and biases
                // Neuron h1
                this.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                this.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                this.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                // Neuron h2
                this.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                this.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                this.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                // Neuron o1
                this.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                this.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                this.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3
                this.result.push(y_pred);

            }
            console.log("In Train Part " + epoch);
            console.log("Neuron one : ");
            console.log("New weight : " + this.w1 + " | " + this.w2);
            console.log("New biais : " + this.b1);
            console.log("Neuron two : ");
            console.log("New weight : " + this.w3 + " | " + this.w4);
            console.log("New biais : " + this.b2);
            console.log("Neuron three : ");
            console.log("New weight : " + this.w5 + " | " + this.w6);
            console.log("New biais : " + this.b3);
        }
        for (var i = 0; i < this.result.length; i += 4) {
            if (i % 50 == 0) {
                var loss = mse_loss(all_y_trues, this.result, i);
                console.log("Epoch " + i + " loss: " + loss);
            }
        }
    }
}

data = [
    [-25, -17],
    [20, 16],
    [3, 4],
    [-15, -10]
  ]
  all_y_trues = [
    1, 
    0, 
    0, 
    1
  ]
  
  // Train our neural network!
  network = new NeuralNetworkLike();
  network.train(data, all_y_trues);

  console.log("---> " + network.feedforward([-12, -17]));
  console.log("---> " + network.feedforward([16, 4]));
