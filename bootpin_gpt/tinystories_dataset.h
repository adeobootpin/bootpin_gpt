void SpinForEver(const char* pszMessage);

struct TrainingExamples
{
	lten::Tensor tokens;
	lten::Tensor target;
	lten::Tensor self_attn_mask;
	uint32_t* ignore_counts;
};


class TinyStoriesDataset : public DataLoader<TrainingExamples>
{
public:
	TinyStoriesDataset() {}
	~TinyStoriesDataset() {}

	bool Init(const char* file_name, int max_context_len, int batch_size)
	{
		bool ret;
		FILE* stream;
		int byte;
		uint64_t count;
		char temp[temp_buffer_size];
		size_t len;
		size_t max_len;
		bool copy;
		uint32_t block_allocate_stride;
		lten::TensorOps options;

		tokenizer_ = InitializeTokenizer("f:\\src\\bootpin_tokenizer\\data\\tokenizer.bin");

		BOS_ = ::GetVocabularySize(tokenizer_);
		EOS_ = BOS_ + 1;
		PAD_ = EOS_ + 1;
		vocabular_size_ = PAD_ + 1;

		options.data_type = lten::INT32;
		batch_ = new TrainingExamples;
		batch_->tokens = lten::AllocateTensor({ (uint64_t)batch_size, (uint64_t)max_context_len }, &options);
		//batch_->target = lten::AllocateTensor({ (uint64_t)batch_size, (uint64_t)max_context_len, 1 }, &options);
		batch_->target = lten::AllocateTensor({ (uint64_t)batch_size, (uint64_t)max_context_len, 1 });
		batch_->self_attn_mask = lten::AllocateTensor({ (uint64_t)batch_size, 1, (uint64_t)max_context_len, (uint64_t)max_context_len });
		batch_->ignore_counts = new uint32_t[batch_size];

		max_context_len_ = max_context_len;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
		errno_t err;
		err = fopen_s(&stream, file_name, "rb");
		if (err)
		{
			ret = false;
			goto Exit;
		}
#else
		stream = fopen(file_name, "rb");
		if (!stream)
		{
			ret = -1;
			goto Exit;
		}
#endif

		text_examples_ = nullptr;
		num_training_examples_ = 0;
		count = 0;
		copy = false;
		max_len = 0;
		memset(temp, 0, sizeof(temp));
		block_allocate_stride = 1000000;

		while ((byte = fgetc(stream)) != EOF)
		{
			if (copy)
			{
				temp[len++] = (char)byte;
				if (len >= sizeof(temp))
				{
					SpinForEver("Buffer too small error loading dataset!\n");
				}
			}

			if ((char)byte == '"')
			{
				if (copy)
				{
					byte = fgetc(stream); // see if the next character is another '"', if it is, then this is not a delimiter but quotes in the text
					if (byte == EOF)
					{
						ret = false;
						goto Exit;
					}
					count++;
					if ((char)byte != '"')
					{
						if (len > max_len)
						{
							max_len = len;
						}

						copy = false;
						temp[len] = '\0';


						if (!(num_training_examples_ % block_allocate_stride))
						{
							text_examples_ = (char**)BlockRealloc(text_examples_, sizeof(char*) * (num_training_examples_), sizeof(char*) * (num_training_examples_ + block_allocate_stride));
						}

						text_examples_[num_training_examples_] = new char[len + 1];
						strcpy_s(text_examples_[num_training_examples_], len + 1, temp);

						num_training_examples_++;
					}
				}
				else
				{
					copy = true;
					len = 0;
				}
			}

			count++;
		}


		DataLoader::Init(batch_size, num_training_examples_, 0, batch_);
		get_item_retries_ = 0;

		ret = true;
	Exit:

		return ret;
	}

	virtual void set_batch_size(uint32_t new_batch_size)
	{
		lten::TensorOps options;
		uint64_t batch_size;

		batch_size = batch_->tokens.get_sizes()[0];

		options.data_type = lten::INT32;

		if (batch_size != new_batch_size)
		{
			batch_->tokens = lten::AllocateTensor({ (uint64_t)new_batch_size, (uint64_t)max_context_len_ }, &options);
			//batch_->target = lten::AllocateTensor({ (uint64_t)new_batch_size, (uint64_t)max_context_len_, 1 }, &options);
			batch_->target = lten::AllocateTensor({ (uint64_t)new_batch_size, (uint64_t)max_context_len_, 1 });
			batch_->self_attn_mask = lten::AllocateTensor({ (uint64_t)new_batch_size, 1, (uint64_t)max_context_len_, (uint64_t)max_context_len_ });
		}
	}


	virtual void get_item(uint64_t training_set_index, uint32_t batch_index)
	{
		uint32_t* tokens;
		float* target;
		float* self_attn_mask;
		uint32_t len;
		uint32_t char_len;
		uint32_t pad_len;
		uint32_t i;
		uint32_t j;
		int ret;


		tokens = (uint32_t*)batch_->tokens.get_data_ptr() + batch_index * max_context_len_;
		target = (float*)batch_->target.get_data_ptr() + batch_index * max_context_len_;
		self_attn_mask = (float*)batch_->self_attn_mask.get_data_ptr() + batch_index * max_context_len_ * max_context_len_;

		while (true)
		{
			char_len = (uint32_t)strlen(text_examples_[training_set_index]);
			if (char_len < 10)
			{
				training_set_index = rand() % num_training_examples_;
				continue;
			}

			tokens[0] = BOS_;
			len = max_context_len_ - 1;
			ret = Encode(tokenizer_, text_examples_[training_set_index], &tokens[1], &len);
			if (ret || (len > char_len))
			{
				training_set_index = rand() % num_training_examples_;
			}
			else
			{
				break;
			}
		}


		for (i = 0; i < len; i++)
		{
			target[i] = tokens[i + 1];
		}
		target[len] = EOS_;


		len++; // accomodate BOS_/EOS_
		pad_len = max_context_len_ - len;
		for (i = 0; i < pad_len; i++)
		{
			tokens[len + i] = PAD_;
			target[len + i] = PAD_;
		}

		batch_->ignore_counts[batch_index] = pad_len;

		//
		// generate self attention mask
		//
		for (i = 0; i < max_context_len_; i++)
		{
			for (j = 0; j < max_context_len_; j++)
			{
				if (i < j)
				{
					self_attn_mask[i * max_context_len_ + j] = 1; // look ahead masking
				}
				else
				{
					if (tokens[j] == PAD_)
					{
						self_attn_mask[i * max_context_len_ + j] = 1;
					}
					else
					{
						self_attn_mask[i * max_context_len_ + j] = 0;
					}
				}
			}
		}

	}


	virtual void swap_examples(uint64_t index_1, uint64_t index_2)
	{

	}

	uint32_t GetVocabularySize()
	{
		return vocabular_size_;
	}

	uint64_t GetNumTrainingExamples()
	{
		return num_training_examples_;
	}

private:
	enum { temp_buffer_size = 10000 };
	uint64_t num_training_examples_;
	char** text_examples_;
	TrainingExamples* batch_;
	uint32_t max_context_len_;
	void* tokenizer_;
	uint32_t BOS_;
	uint32_t EOS_;
	uint32_t PAD_;
	uint32_t vocabular_size_;
	uint64_t get_item_retries_;
};

