#include <bits/stdc++.h>
#include "Eigen/Dense"
#include <string.h>

#define float  double
using namespace Eigen;
using namespace std;

// weight values.
struct params{
	Eigen :: RowVectorXd w;
	float w0;
};

// data structure for holding data features and target class
struct data_structure{
	Eigen::MatrixXd features;
	Eigen :: RowVectorXd target_class;
};

data_structure read_data(string path)
{
	ifstream trainFile;
	trainFile.open(file_location.c_str());
	vector<vector<float> > train_data;
	vector<int> target_class;
	f1 = fopen(path);
	if(trainFile.is_open())
	{
		while(!trainFile.eof())
		{
			string s;
			vector<float> temp;
			getline(trainFile, s);
			int comma_no = 0;
			int prev = 0;
			for(int i=0;i<s.length(); i++)
			{
				if(s[i]==',')
				{
					string temp_float = s.substr(prev,i-prev);
					prev = i+1;
					temp.push_back(stof(temp_float, NULL));
				}
			}
			string ss = s.substr(prev,s.length()-prev);
			int c = stoi(ss);
			target_class.push_back(c);
			train_data.push_back(temp);
		}
	}

	Eigen::MatrixXd train_data_eig(train_data.size(),train_data[0].size());
	for(int i=0;i<train_data.size();i++)
        for(int j=0;j<train_data[0].size();j++)
            train_data_eig(i,j) = train_data[i][j];

    Eigen :: RowVectorXd target_class_eig(target_class.size());
    for(int j=0;j<target_class.size();j++)
            target_class_eig(j) = target_class[j];

    data_structure d;
	d.features = train_data_eig;
	d.target_class = target_class_eig;
	return d;
}

vector<params> build_nn(int input_layer_size,int hidden_layer_size,int output_layers)
{
	vector<params> weight_mat(2);
	for(int i = 0;i<input_layer_size*hidden_layer_size;i++)
		weight_mat[0].w<<((double) rand() / (RAND_MAX))

	for(int i = 0;i<output_layers*hidden_layer_size;i++)
		weight_mat[1].w<<((double) rand() / (RAND_MAX))
	
	weight_mat[0].w0=((double) rand() / (RAND_MAX));
	weight_mat[1].w0=((double) rand() / (RAND_MAX));
	
	return weight_mat
}



int main()
{
	Eigen::MatrixXd train_data_eig(3100,64);
	train_data_eig = read_data('train.txt');

	int hidden_layer_size  = 5;
	int output_layers = 10;
	vector<param> weight_matrix = build_nn(train_data_eig.rows(),hidden_layer_size,output_layers);

	Eigen::MatrixXd validation_data_eig(3100,64);
	train_data_eig = read_data('validation.txt');
	
	nn_compile(train_data_eig,hidden_layer_size,output_layers,weight_matrix,current_epoch_number);




	return 0;
}