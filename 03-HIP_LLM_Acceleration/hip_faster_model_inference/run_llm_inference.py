import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


prefix_str = '''
continue to write this story.
The sun rose thinly from the sea and the old man could see the other boats, low on the water and well in toward the shore, spread out across the current. Then the sun was brighter and the glare came on the water and then, as it rose clear, the flat sea sent it back at his eyes so that it hurt sharply and he rowed without looking into it. He looked down into the water and watched the lines that went straight down into the dark of the water. He kept them straighter than anyone did, so that at each level in the darkness of the stream there would be a bait waiting exactly where he wished it to be for any fish that swam there. Others let them drift with the current and sometimes they were at sixty fathoms when the fishermen thought they were at a hundred.
But, he thought, I keep them with precision. Only I have no luck any more. But who knows? Maybe today. Every day is a new day. It is better to be lucky. But I would rather be exact. Then when luck comes you are ready.
The sun was two hours higher now and it did not hurt his eyes so much to look into the east. There were only three boats in sight now and they showed very low and far inshore.
All my life the early sun has hurt my eyes, he thought. Yet they are still good. In the evening I can look straight into it without getting the blackness. It has more force in the evening too. But in the morning it is painful.
Just then he saw a man-of-war bird with his long black wings circling in the sky ahead of him. He made a quick drop, slanting down on his back-swept wings, and then circled again.
"He's got something," the old man said aloud. "He's not just looking."
He rowed slowly and steadily toward where the bird was circling. He did not hurry and he kept his lines straight up and down. But he crowded the current a little so that he was still fishing correctly though faster than he would have fished if he was not trying to use the bird.
The bird went higher in the air and circled again, his wings motionless. Then he dove suddenly and the old man saw flying fish spurt out of the water and sail desperately over the surface.
"Dolphin," the old man said aloud. "Big dolphin."
He shipped his oars and brought a small line from under the bow. It had a wire leader and a medium-sized hook and he baited it with one of the sardines. He let it go over the side and then made it fast to a ring bolt in the stern. Then he baited another line and left it coiled in the shade of the bow. He went back to rowing and to watching the long-winged black bird who was working, now, low over the water.
As he watched the bird dipped again slanting his wings for the dive and then swinging them wildly and ineffectually as he followed the flying fish. The old man could see the slight bulge in the water that the big dolphin raised as they followed the escaping fish. The dolphin were cutting through the water below the flight of the fish and would be in the water, driving at speed, when the fish dropped. It is a big school of dolphin, he thought. They are widespread and the flying fish have little chance. The bird has no chance. The flying fish are too big for him and they go too fast.
He watched the flying fish burst out again and again and the ineffectual movements of the bird. That school has gotten away from me, he thought. They are moving out too fast and too far. But perhaps I will pick up a stray and perhaps my big fish is around them. My big fish must be somewhere.
'''

Configs = [
    # batch, input_length, output_length
    (1, 128, 8),
    (1, 256, 16),
    (1, 512, 32),
    #(1, 1024, 64),
]


def main(model_name, fastgemv=False, gpu=0):
    torch.cuda.empty_cache()
    print('\nModel: ', model_name, '\nOpt: ', fastgemv, '\n')

    if torch.cuda.is_available():
        device = torch.device("cuda:{}".format(gpu))
    else:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code = True, fastgemv = fastgemv).half().to(device)
    model.pre_treatment()

    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    warmup, freq = 5, 10

    print("batch  query_length  answer_length  query_latency(ms)  answer_latency(ms)  total_latency(ms)  1-token_output_latency(ms)  tokens/second")
    for batch, input_length, output_length in Configs:
        torch.cuda.empty_cache()
        torch.cuda.nvtx.range_push('{} batch {} input_length {} output_length {}'.format(model_name, batch, input_length, output_length))
        # prepare inputs
        inputs = [prefix_str] * batch
        input_ids = tokenizer(inputs, return_tensors='pt').input_ids[:, :input_length].to(device)
        max_length = input_length + output_length

        # warm up
        for _ in range(warmup):
            logits = model.generate(input_ids, num_beams=1, max_length=max_length, use_cache=True)

        # print(logits[0][input_length:])

        st.record()
        for _ in range(freq):
            logits = model.generate(input_ids, num_beams=1, max_length=input_length+1, use_cache=True)
        ed.record()
        ed.synchronize()
        query_latency = st.elapsed_time(ed) / freq

        st.record()
        for _ in range(freq):
            logits = model.generate(input_ids, num_beams=1, max_length=input_length+output_length+1, use_cache=True)
        ed.record()
        ed.synchronize()
        total_latency = st.elapsed_time(ed) / freq
        torch.cuda.nvtx.range_pop()

        answer_lantency = total_latency - query_latency
        token_output_latency = answer_lantency / output_length
        tokens_per_second = (1000 / token_output_latency) * batch

        outputs = tokenizer.batch_decode(logits)
        print(outputs)
        print(str(batch).ljust(len('batch')) + "  " +
                str(input_length).ljust(len('query_length')) + "  " +
                str(output_length).ljust(len('answer_length')) + "  " +
                "{:.3f}".format(query_latency).ljust(len('query_latency(ms)')) + "  " +
                "{:.3f}".format(answer_lantency).ljust(len('answer_latency(ms)')) +  "  " +
                "{:.3f}".format(total_latency).ljust(len('total_latency(ms)')) + "  " +
                "{:.3f}".format(token_output_latency).ljust(len('1-token_output_latency(ms)')) + "  " +
                "{:.3f}".format(tokens_per_second).ljust(len('tokens_second'))) 


if __name__ == "__main__":
    main('bigscience/bloom-7b1', False)
    main('bigscience/bloom-7b1', True)

    main('facebook/opt-6.7b', False)
    main('facebook/opt-6.7b', True)

    main('meta-llama/Llama-2-7b-hf', False)
    main('meta-llama/Llama-2-7b-hf', True)
