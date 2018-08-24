#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include "dart/collision/bullet/bullet.hpp"
#include "dart/collision/ode/ode.hpp"
#include <dart/utils/utils.hpp>
#include <iostream>
#include <fstream>
#include "tiny_dnn/tiny_dnn.h"
#include <string>
#include <typeinfo>

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart;
using namespace utils;
using namespace std;

using namespace tiny_dnn;
using namespace tiny_dnn::activation;
using namespace tiny_dnn::layers;

//const unsigned int testSize = 61884;// 127008

class MyWindow : public dart::gui::SimWindow
{
public:
    
    MyWindow(const WorldPtr& world)
    {
        setWorld(world);
    }
    
    void keyboard(unsigned char key, int x, int y) override
    {
        switch(key)
        {
            default:
                SimWindow::keyboard(key, x, y);
        }
    }
    
    void drawWorld() const override
    {
        // Make sure lighting is turned on and that polygons get filled in
        glEnable(GL_LIGHTING);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        
        SimWindow::drawWorld();
    }
    
    void timeStepping() override
    {
        SimWindow::timeStepping();
    }
protected:
};

vector<vector<double>> extract_bias(const char* file_name)
{
    //Open the file and store all the data
    vector<vector<string>> total_raw_data;
    //vector<vector<vector<float>>> layer_data_mat;
    vector<vector<double>> layer_data_mat; //without neuron separation
    ifstream data_file(file_name);
    string line;
    if (data_file.is_open()) {
        vector<string> layer_raw_data;
        while(getline(data_file, line, '\n')) {  //line breaker separate each layer
            line.erase(remove(line.begin(), line.end(), '\"'), line.end());
            line.erase(remove(line.begin(), line.end(), '['), line.end());
            line.erase(remove(line.begin(), line.end(), ']'), line.end());
            stringstream lineStream(line); //Each line contains all the weights of one layer
            string cell;
            layer_raw_data.clear();
            while(getline(lineStream, cell, ',')) { //Each cell contains the weights mapping from one neuron to next layer
                layer_raw_data.push_back(cell);  // Comma separate each neuron
            }
            total_raw_data.push_back(layer_raw_data); //This contains weights of all layers
        }
        data_file.close();
    } else {
        cout<<"Unable to open file"<<endl;}

    //Change all data from string to float, and store in layer_data_mat
    for (int i = 0; i < total_raw_data.size(); i++) {
        //Get the layer of raw data
        vector<string> curr_raw_layer = total_raw_data[i];
        //The layer after transfered
        vector<double> curr_trans_layer;
        curr_trans_layer.clear();
        //Parsing each <32> in total 5*32
        for (int j = 0; j < curr_raw_layer.size(); j++) {
            double f;      
            stringstream ss(curr_raw_layer[j]);
            while (ss >> f) curr_trans_layer.push_back(f);
        }

        layer_data_mat.push_back(curr_trans_layer);
    }

    // cout<<layer_data_mat.size()<<endl;
    // for (int i = 0; i < layer_data_mat.size(); i++) {
    //     vector<float> curr = layer_data_mat[i];
    //     cout<<curr.size()<<endl;
    //     for(int j = 0; j < curr.size(); j++) {
    //         cout<<curr[j]<<' ';
    //     }
    //     cout<<' '<<endl;
    // }

    return layer_data_mat;
}

vector<vector<double>> extract_weights(const char* file_name, bool is_test)
{   
    //Open the file and store all the data
    vector<vector<string>> total_raw_data;
    //vector<vector<vector<float>>> layer_data_mat;
    vector<vector<double>> layer_data_mat; //without neuron separation
    ifstream data_file(file_name);
    string line;
    if (data_file.is_open()) {
        vector<string> layer_raw_data;
        while(getline(data_file, line, '\n')) {  //line breaker separate each layer
            line.erase(remove(line.begin(), line.end(), '\"'), line.end());
            line.erase(remove(line.begin(), line.end(), '['), line.end());
            line.erase(remove(line.begin(), line.end(), ']'), line.end());
            stringstream lineStream(line); //Each line contains all the weights of one layer
            string cell;
            layer_raw_data.clear();
            while(getline(lineStream, cell, ',')) { //Each cell contains the weights mapping from one neuron to next layer
                layer_raw_data.push_back(cell);  // Comma separate each neuron
            }
            //Space separate data within one neuron set, which is from this nueron targeting to next
            if (!is_test)
            layer_raw_data.erase(layer_raw_data.begin() + layer_raw_data.size()-1); //For file IO reason there is always an extra line
            //If this gets fixed it will be perfect. lol. 
            total_raw_data.push_back(layer_raw_data); //This contains weights of all layers
        }
        data_file.close();

    } else {
        cout<<"Unable to open file"<<endl;}

    //Change all data from string to float, and store in layer_data_mat
    for (int i = 0; i < total_raw_data.size(); i++) {
        //Get the layer of raw data
        vector<string> curr_raw_layer = total_raw_data[i];
        //The layer after transfered
        //vector<vector<float>> curr_trans_layer;
        vector<double> curr_trans_layer; //without neuron separation
        curr_trans_layer.clear();
        //Parsing each <32> in total 5*32
        for (int j = 0; j < curr_raw_layer.size(); j++) {
            double f;      
            stringstream ss(curr_raw_layer[j]);
            //vector<float> one_neuron;
            //while(ss >> f) one_neuron.push_back(f);
            //curr_trans_layer.push_back(one_neuron);
            while (ss >> f) curr_trans_layer.push_back(f);
        }
        layer_data_mat.push_back(curr_trans_layer);
    }

    // cout<<"debugging info!"<<endl;
    // //cout<<layer_data_mat.size()<<endl;
    // cout<<"printing first layer info, should be 5*32"<<endl;
    // for (int i = 0; i < layer_data_mat.size(); i++) {
    //     for (int j = 0; j < layer_data_mat[i].size(); j++) {
    //         cout<<layer_data_mat[i][j]<<' ';
    //     }
    //     cout<<' '<<endl;
    // }
    return layer_data_mat;
}

network<sequential> network_construct(vector<vector<double>> weights_mat, vector<vector<double>> bias_mat, bool is_classifier)
{
        //Initialize neural net based on type. classifier vs regressor
    network<sequential> net;
    if (is_classifier == true) {
        cout<<"Classifier net!"<<endl;
        net << fully_connected_layer(9, 128) << activation::tanh()
        << fully_connected_layer(128, 128) << activation::tanh()
        << fully_connected_layer(128, 64) << activation::tanh()
        << fully_connected_layer(64, 3) << activation::softmax();
        assert(net.in_data_size() == 9);
        assert(net.out_data_size() == 3);
    } else {
        cout<<"Regression net!"<<endl;
        net << fully_connected_layer(9, 128) << activation::relu()
        << fully_connected_layer(128, 128) << activation::relu()
        << fully_connected_layer(128, 64) << activation::relu()
        << fully_connected_layer(64, 3);
        assert(net.in_data_size() == 9);
        assert(net.out_data_size() == 3);
    }

    layer* l;
    int l_cnt = 0;
    for (int n = 0; n < net.layer_size(); n++) {
        l = net[n];
        if (l->layer_type() == "fully-connected") {
            auto info = l->weights();
            vec_t &w = *(info[0]);
            vec_t &b = *(info[1]);
            //float* first_ptr = &layer_data_mat[l_cnt][0].front(); //with neuron vector separation
            double* w_fpt = &weights_mat[l_cnt].front(); //without neuron vector separation
            double* b_fpt = &bias_mat[l_cnt].front();
            w.assign(w_fpt, w_fpt + weights_mat[l_cnt].size());
            b.assign(b_fpt, b_fpt + bias_mat[l_cnt].size());
            l_cnt += 1;
        }   
    }
    net.save("rect_r3_sym");
    return net;
}


void evaluateNet(network<sequential> net, vector<vec_t> test_data, vector<vec_t> test_target)
{
    double loss = net.get_loss<mse>(test_data, test_target);
    cout<<loss<<endl;
}

vector<vec_t> vector_transfer(vector<vector<double>> myVec)
{
    vector<vec_t> result;
    for (int i = 0; i < myVec.size(); i++)
    {
        vec_t temp;
        vector<double> curr = myVec[i];
        double* pt = &curr.front();
        temp.assign(pt, pt + curr.size());
        result.push_back(temp);
    }
    return result;

}

int main(int argc, char* argv[])
{

    WorldPtr world = SkelParser::readWorld(DART_DATA_PATH"/NN-contact-force/skel/singleBody_genData.skel");
    assert(world != nullptr);
    world->setGravity(Eigen::Vector3d(0.0, 0.0, 0));
    
    string temp ="rect_r3_sym_3D_weights.csv";
    string temp_2 ="rect_r3_sym_3D_bias.csv";
    const char* weight_name;
    const char* bias_name;
    weight_name = temp.c_str();
    bias_name = temp_2.c_str();
    int flag = -1;
    bool is_classifier;
    cout<<"Type 0 for regressor; 1 for classifier "<<endl;
    cin>>flag;
    is_classifier = (flag == 1) ? (true):(false); 

    //construct_mlp_from_weights(path_name, is_classifier);
    vector<vector<double>> weights_mat = extract_weights(weight_name, false);
    vector<vector<double>> bias_mat = extract_bias(bias_name);
    network<sequential> net = network_construct(weights_mat, bias_mat, is_classifier);
    cout<<"Transfer Finished!"<<endl;

    // const char* x_test = "";
    // const char* y_test = "";

    // vector<vector<double>>  input = extract_weights("/home/jsun303/Desktop/dart-NN-contact-force/data/NN-contact-force/test_data/c1_x_test.csv", true);
    // vector<vector<double>>  output = extract_bias("/home/jsun303/Desktop/dart-NN-contact-force/data/NN-contact-force/test_data/c1_y_test.csv");

    // vector<vec_t> x_test = vector_transfer(input);
    // vector<vec_t> y_test = vector_transfer(output);
    // evaluateNet(net, x_test, y_test);


    //Testing if network models are successfully transfered by testing specific data set

    // vector<vector<double>> x_test;
    // double first_arr[] = {0.510521, -0.859865,  -2.08544,   -6.72442,   -2.32748};
    // vector<double> first_vec (first_arr, first_arr + sizeof(first_arr) / sizeof(double));
    // x_test.push_back(first_vec);

    // double second_arr[] = {-0.328948,   -0.944348,  -0.0272732, 9.12826,    -4.19388};
    // vector<double> second_vec (second_arr, second_arr + sizeof(second_arr) / sizeof(double));
    // x_test.push_back(second_vec);

    // double third_arr[] = {0.16081,  0.986985,   3.58803,    -9.08075,   -0.740867};
    // vector<double> third_vec (third_arr, third_arr + sizeof(third_arr) / sizeof(double));
    // x_test.push_back(third_vec);

    // double fourth_arr[] = {-0.132366,   -0.991201,  2.15397,    7.04784,    0.526174};
    // vector<double> fourth_vec (fourth_arr, fourth_arr + sizeof(fourth_arr) / sizeof(double));
    // x_test.push_back(fourth_vec);

    // double fifth_arr[] = {0.272638, -0.962117,  3.54762,    7.57354,    -4.66197};
    // vector<double> fifth_vec (fifth_arr, fifth_arr + sizeof(fifth_arr) / sizeof(double));
    // x_test.push_back(fifth_vec);

    // for (int k = 0; k < 5; k++)
    // {
    //     vec_t in;
    //     double* test_pr = &x_test[k].front();
    //     in.assign(test_pr, test_pr + x_test[k].size());

    //     vec_t result = net.predict(in);
    //     for (int i = 0; i < result.size(); i++) {
    //         cout<<result[i]<<' ';
    //     }
    //     cout<<' '<<endl;
    // }

    // cout<<"Debugging!"<<endl;
    // cout<<' '<<endl;
    // layer* h;
    // for (int n = 0; n < net.layer_size(); n++) {
    //     h = net[n];
    //     if (h->layer_type() == "fully-connected") {
    //         cout<<"Current at layer "<<n<<endl;
    //         auto info = h->weights();
    //         vec_t &w = *(info[0]);
    //         vec_t &b = *(info[1]);
    //         for (int k = 0; k < w.size(); k++) {
    //             cout<<w[k]<<' ';
    //         }
    //         cout<<' '<<endl;
    //     }
    // }

}
