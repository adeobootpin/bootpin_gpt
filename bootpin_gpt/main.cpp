#include <stdio.h>
#include <stdint.h>
#include <filesystem>
#include <fstream>
#include <assert.h>
#include <errno.h>

#include "tensor.h"
#include "lten.h"
#include "threadpool2.h"
#include "data_loader.h"
#include "bootpin_tokenizer.h"

#include "tinystories_dataset.h"


void PrintETA(double nseconds_latest_iteration, uint32_t remaining_iterations);
void Test(uint32_t max_seq_len, uint16_t num_heads, uint32_t hidden_dim);

struct AttentionBlock : public lten::Module
{
public:
	AttentionBlock(int hidden_dim, int num_heads, int max_seq_len, float drop_out_rate, lten::NeuralNetwork* net)
	{
		lnorm1_ = net->register_module("attn_norm1", lten::LayerNorm(hidden_dim, true));
		mha_ = net->register_module("attn_mha", lten::MultiheadAttention(hidden_dim, num_heads, true));
		lnorm2_ = net->register_module("attn_norm2", lten::LayerNorm(hidden_dim, true));

		mlp_fc1_ = net->register_module("attn_mlp_fc1", lten::FullyConnected(hidden_dim, hidden_dim, true));
		mlp_drop_out1_ = net->register_module("attn_mlp_drop_out1", lten::Dropout(drop_out_rate));
		mlp_fc2_ = net->register_module("attn_mlp_fc2", lten::FullyConnected(hidden_dim, hidden_dim, true));
		mlp_drop_out2_ = net->register_module("attn_mlp_drop_out2", lten::Dropout(drop_out_rate));
	}

	lten::Tensor mlp(lten::Tensor x)
	{
		x = mlp_fc1_->forward(x);
		x = lten::gelu(x);
		x = mlp_drop_out1_->forward(x);
		x = mlp_fc2_->forward(x);
		x = mlp_drop_out2_->forward(x);
		return x;
	}

	lten::Tensor forward(lten::Tensor x, lten::Tensor mask)
	{
		lten::Tensor x_res;

		x_res = lnorm1_->forward(x);
		x = x + mha_->forward(x_res, &mask);
		x = x + mlp(lnorm2_->forward(x));
		return x;
	}

	bool init() { return true; }

	void to(const lten::device device, int target_device_index = 0)
	{
		lnorm1_->to(device, target_device_index);
		mha_->to(device, target_device_index);
		lnorm2_->to(device, target_device_index);
		mlp_fc1_->to(device, target_device_index);
		mlp_fc2_->to(device, target_device_index);
		mlp_drop_out1_->to(device, target_device_index);
		mlp_drop_out2_->to(device, target_device_index);
	}

	void train(bool on)
	{
		lnorm1_->train(on);
		mha_->train(on);
		lnorm2_->train(on);
		mlp_fc1_->train(on);
		mlp_fc2_->train(on);
		mlp_drop_out1_->train(on);
		mlp_drop_out2_->train(on);
	}


	std::vector<lten::Tensor*> get_all_weights()
	{
		std::vector<lten::Tensor*> weights;
		std::vector<lten::Tensor*> sub_module_weights;
		
		return weights;
	}

	void clear_gradients()
	{
		assert(0); // is this ever called? this function can probably be left empty
		lnorm1_->clear_gradients();
		mha_->clear_gradients();
		lnorm2_->clear_gradients();
		mlp_fc1_->clear_gradients();
		mlp_fc2_->clear_gradients();
	}

private:
	lten::LayerNorm* lnorm1_;
	lten::MultiheadAttention* mha_;
	lten::LayerNorm* lnorm2_;

	lten::FullyConnected* mlp_fc1_;
	lten::Dropout* mlp_drop_out1_;
	lten::FullyConnected* mlp_fc2_;
	lten::Dropout* mlp_drop_out2_;
};


class Net : public lten::NeuralNetwork
{
public:
	Net(uint32_t vocab_size, uint32_t max_seq_len, uint16_t num_heads, uint32_t hidden_dim, float drop_out_rate)
	{
		uint32_t i;
		lten::TensorOps options;

		tok_embeddings_ = register_module("tok_embeddings", lten::Embedding(vocab_size, hidden_dim));
		pos_embeddings_ = register_module("pos_embedding", lten::Embedding(max_seq_len, hidden_dim));

		for (i = 0; i < total_blocks; i++)
		{
			attn_block[i] = register_module("attn", AttentionBlock(hidden_dim, num_heads, max_seq_len, drop_out_rate, this));
		}

		lnorm_ = register_module("lnorm", lten::LayerNorm(hidden_dim, true));
		proj_= register_module("proj", lten::FullyConnected(hidden_dim, vocab_size, true));

		int* temp = new int[max_seq_len];

		for (i = 0; i < max_seq_len; i++)
		{
			temp[i] = i;
		}
		
		options.data_type = lten::INT32;
		pos_ = lten::TensorFromBuffer({ 1, max_seq_len }, temp, true, &options);

	}

	lten::Tensor forward(lten::Tensor x, lten::Tensor mask)
	{
		int i;
		lten::Tensor tok_emb;
		lten::Tensor pos_emb;
		
		tok_emb = tok_embeddings_->forward(x);
		pos_emb = pos_embeddings_->forward(pos_);

		x = tok_emb + pos_emb;

		for (i = 0; i < total_blocks; i++)
		{
			x = attn_block[i]->forward(x, mask);
		}

		x = lnorm_->forward(x);
		x = proj_->forward(x);

		return x;
	}



	void to(const lten::device device, int target_device_index = 0)
	{
		pos_ = pos_.to(device, target_device_index);
		NeuralNetwork::to(device, target_device_index);

		/*
		int i;

		tok_embeddings_->to(device, target_device_index);
		pos_embeddings_->to(device, target_device_index);
		lnorm_->to(device, target_device_index);
		proj_->to(device, target_device_index);
		pos_ = pos_.to(device, target_device_index);

		for (i = 0; i < total_blocks; i++)
		{
			attn_block[i]->to(device, target_device_index);
		}
		*/
	}

	void train(bool on)
	{
		int i;

		tok_embeddings_->train(on);
		pos_embeddings_->train(on);
		lnorm_->train(on);
		proj_->train(on);

		for (i = 0; i < total_blocks; i++)
		{
			attn_block[i]->train(on);
		}
	}

private:
	enum { total_blocks = 6 };

	lten::Embedding* tok_embeddings_;
	lten::Embedding* pos_embeddings_;
	AttentionBlock* attn_block[total_blocks];
	lten::LayerNorm* lnorm_;
	lten::FullyConnected* proj_;
	lten::Tensor pos_;

};


int main()
{
	//return gpt_llama2();


	TrainingExamples* batch;
	uint32_t batch_size = 12;
	uint32_t max_seq_len = 512;
	uint16_t num_heads = 8;
	uint32_t hidden_dim = 768;
	//hidden_dim = 512;
	uint32_t num_epochs = 2;
	//hidden_dim = 512;
	num_heads = 8;
	//batch_size = 4;
	lten::device dev = lten::GPU;


	//Test(max_seq_len, num_heads, hidden_dim); return 0;

	lten::Tensor x;
	lten::Tensor target;
	lten::Tensor self_attn_mask;
	lten::Tensor logits;
	lten::Tensor probs;
	lten::Tensor loss;
	float loss_val = 0;
	uint64_t estimated_total_iterations;
	uint64_t total_iterations = 0;
	uint64_t avg_index = 0;
	uint32_t ignore_counts;
	int gradient_accumulation_steps;

	std::chrono::steady_clock::time_point clock_begin;
	std::chrono::steady_clock::time_point clock_end;
	std::chrono::steady_clock::duration time_span;
	double nseconds;
	double nseconds_total = 0;

	TinyStoriesDataset ds;
	const char* file_name = "c:\\src\\bootpin_gpt\\data\\training\\TinyStories.csv";
	//const char* file_name = "e:\\src\\bootpin_gpt\\data\\training\\titch.csv";
	//const char* file_name = "f:\\src\\bootpin_gpt\\data\\training\\titch.csv"; max_seq_len = 32; batch_size = 4; num_heads = 1;

	ds.Init(file_name, max_seq_len, batch_size);

	Net net(ds.GetVocabularySize(), max_seq_len, num_heads, hidden_dim, 0.0f);

	net.load_checkpoint("c:\\src\\bootpin_gpt\\data\\bootpin_gpt.bin");

	lten::AdamOptimizer optimizer;
	optimizer.attach_network(net);
	//optimizer.set_learning_rate((float)3e-4);
	optimizer.set_learning_rate((float)5e-5);


	net.to(dev);
	net.train(true);

	uint64_t param_count;
	net.get_statistics(nullptr, &param_count);
	printf("Number of parameters: %lld\n", param_count);

	gradient_accumulation_steps = 120 / batch_size;

	estimated_total_iterations = num_epochs * ds.GetNumTrainingExamples() / batch_size; // estimated because iterations do not always have full batch size


	for (uint32_t epoch = 0; epoch < num_epochs; epoch++)
	{
		ds.Prefetch();

		while (batch = ds.GetBatch())
		{
			clock_begin = std::chrono::steady_clock::now();

			x = batch->tokens.to(dev);
			target = batch->target.to(dev);
			self_attn_mask = batch->self_attn_mask.to(dev);

			batch_size = (int)x.get_sizes()[0];
			ignore_counts = 0;
			for (uint32_t j = 0; j < batch_size; j++)
			{
				ignore_counts += batch->ignore_counts[j];
			}

			if (dev == lten::GPU)
			{
				ds.Prefetch();
			}

			logits = net.forward(x, self_attn_mask);

			probs = log_softmax(logits, 2);

			if (dev == lten::CPU)
			{
				ds.Prefetch();
			}

			loss = nll_loss(probs, target, 2, 10258, ignore_counts);
			loss = loss * (1.0f / gradient_accumulation_steps);
			loss.backward();


			total_iterations++;

			if (!(total_iterations % gradient_accumulation_steps))
			{
				lten::Tensor temp = loss.to(lten::CPU);
				float floss = (*((float*)temp.get_data_ptr())) * gradient_accumulation_steps;
				loss_val += floss;
				printf("loss: %f avg loss: %f [epoch: %d iteration: %lld / %lld]\n", floss, loss_val / (avg_index + 1), epoch, total_iterations, estimated_total_iterations);
				avg_index++;

				optimizer.step();
				optimizer.zero_grad();

				PrintETA(nseconds_total / total_iterations, estimated_total_iterations - total_iterations);
				printf("\n");
			}

			if (!(total_iterations % 500))
			{
				//net.save_checkpoint("e:\\src\\bootpin_gpt\\data\\bootpin_gpt.bin");
				//net.save_checkpoint("c:\\src\\bootpin_gpt\\data\\titch.bin");
			}

			if (!(total_iterations % 50000))
			{
				char filename[256];
				sprintf_s(filename, sizeof(filename), "c:\\src\\bootpin_gpt\\data\\bootpin_gpt_ex4_%lld.bin", total_iterations);
				net.save_checkpoint(filename);
			}


			clock_end = std::chrono::steady_clock::now();
			time_span = clock_end - clock_begin;
			nseconds = double(time_span.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
			nseconds_total += nseconds;
		}
	}


	net.save_checkpoint("c:\\src\\bootpin_gpt\\data\\bootpin_gpt.bin");
	//net.save_checkpoint("c:\\src\\bootpin_gpt\\data\\titch.bin");

	Test(max_seq_len, num_heads, hidden_dim); return 0;
	return 0;
}




uint32_t tokens[1024];
void Test(uint32_t max_seq_len, uint16_t num_heads, uint32_t hidden_dim)
{
	int ret;
	uint32_t i;
	uint32_t j;
	uint32_t pad_len;
	uint32_t BOS;
	uint32_t EOS;
	uint32_t PAD;
	//uint32_t tokens[1024];
	float* self_attn_mask;

	lten::Tensor x;
	lten::Tensor mask;
	lten::Tensor x_nn;
	lten::Tensor mask_nn;
	lten::TensorOps options;
	lten::Tensor logits;
	lten::Tensor probs;
	uint32_t next_token;
	uint32_t total_tokens;

	uint32_t len;



	lten::device dev = lten::GPU;

	uint32_t vocabular_size;
	void* tokenizer;

	tokenizer = InitializeTokenizer("f:\\src\\bootpin_tokenizer\\data\\tokenizer.bin");

	vocabular_size = GetVocabularySize(tokenizer);

	BOS = vocabular_size;
	EOS = BOS + 1;
	PAD = EOS + 1;
	vocabular_size = PAD + 1;

	Net net(vocabular_size, max_seq_len, num_heads, hidden_dim, 0);

	ret = net.load_checkpoint("f:\\src\\bootpin_gpt\\data\\bootpin_gpt.bin");

	net.to(dev);
	net.train(false);


	tokens[0] = BOS;
	len = sizeof(tokens) / sizeof(tokens[0]) - 1;
	//ret = Encode(tokenizer, "", &tokens[1], &len);
	//ret = Encode(tokenizer, "Tom and Jane are friends. One day, Jane goes to Tom's house. Tom has a big pot of soup. He wants to share it with Jane. \"Jane, do you want some soup?\" Tom asks.", &tokens[1], &len);
	ret = Encode(tokenizer, "Once upon a time, Tom and Jane went to the park.", &tokens[1], &len);
	//ret = Encode(tokenizer, "", &tokens[1], &len);


	len++; // accomodate BOS
	pad_len = max_seq_len - len;
	for (i = 0; i < pad_len; i++)
	{
		tokens[len + i] = PAD;
	}

	self_attn_mask = new float[max_seq_len * max_seq_len];

	options.data_type = lten::INT32;
	x = lten::TensorFromBuffer({ (uint64_t)1, (uint64_t)max_seq_len }, tokens, false, &options);
	mask = lten::TensorFromBuffer({ (uint64_t)1, 1, (uint64_t)max_seq_len, (uint64_t)max_seq_len }, self_attn_mask, false);

	total_tokens = len;
	while (true)
	{
		//
		// generate self attention mask
		//
		for (i = 0; i < max_seq_len; i++)
		{
			for (j = 0; j < max_seq_len; j++)
			{
				if (i < j)
				{
					self_attn_mask[i * max_seq_len + j] = 1; // look ahead masking
				}
				else
				{
					if (tokens[j] == PAD)
					{
						self_attn_mask[i * max_seq_len + j] = 1;
					}
					else
					{
						self_attn_mask[i * max_seq_len + j] = 0;
					}
				}
			}
		}

		x_nn = x.to(dev);
		mask_nn = mask.to(dev);

		logits = net.forward(x_nn, mask_nn);
		probs = softmax(logits, -1);

		probs = probs.squeeze(0);
		probs = probs.to(lten::CPU);

		lten::Tensor val = lten::Multinomial(probs, 1);


		float* debug = (float*)probs.get_data_ptr();
		debug += (total_tokens - 1) * vocabular_size;
		float max_val = -1;
		int max_idx = 0;
		for (j = 0; j < vocabular_size; j++)
		{
			if (debug[j] > max_val)
			{
				max_val = debug[j];
				max_idx = j;
			}
		}



		//next_token = (uint32_t)((float*)val.get_data_ptr())[total_tokens-1];
		next_token = max_idx;

		tokens[total_tokens] = next_token;

		if (next_token == EOS)
		{
			break;
		}

		if (total_tokens >= max_seq_len)
		{
			break;
		}
		total_tokens++;

		//unsigned int w_buffer_len = 5000;
		//wchar_t w_buffer[5000];
		//Decode(tokenizer, &tokens[1], total_tokens - 1, w_buffer, &w_buffer_len);
		//std::wcout << w_buffer << std::endl;
	}


	unsigned int w_buffer_len = 10000;
	wchar_t* w_buffer = new wchar_t[w_buffer_len];

	Decode(tokenizer, &tokens[1], total_tokens - 1, w_buffer, &w_buffer_len);
	std::wcout << w_buffer << std::endl;

	printf("Len: %d\n", (int)wcslen(w_buffer));
	printf("Tokens: %d\n", total_tokens);

	delete w_buffer;

	while (true)
	{

	}

}


void SpinForEver(const char* pszMessage)
{
	while (true)
	{
		printf("\r\n%s", pszMessage);
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
}


