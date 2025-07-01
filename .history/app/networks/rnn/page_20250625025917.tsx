"use client";

import { MathJax } from "better-react-mathjax";
import ContentLayout from "@/app/ui/ContentLayout";

export default function Page() {
  const intro = (
    <p>
      Recurrent neural networks formed the basis of many early language models. Their sequential structure makes them well-suited for formal connections to classical models of computation, resulting in a rich field of study of their representational capacity.
    </p>
  );

  const sections = [
    {
      title: "Recurrent Neural Networks and Language Models",
      content: (
        <>
          <p>
            Recurrent neural LMs are LMs whose conditional probabilities are given by a recurrent neural network. Most results pertain to <strong>Elman RNNs</strong> as they are the easiest to analyze and are special cases of more common networks, e.g., LSTMs and GRUs. LSTMs are also theoretically more powerful in some sense.
          </p>
          <h4 className="mt-6 mb-2 font-semibold">Definition</h4>
          <div className="flex justify-center my-4">
            <MathJax>{`
              $$
              \\text{Elman RNN:} \\\\
             \\rnn = \\elmanrnntuple \\\\
              \\text{with hidden state recurrence:} \\\\
              \\begin{align}
                &\\hiddStatetzero = \\rnnInitstate (t=0) \\\\
                &\\hiddStatet = \\activation\\left(\\recMtx \, \\vhtminus + \\inMtx \, \\inEmbedding(\\symt) + \\biasVec \\right) (t>0)
              \\end{align}
              $$
            `}</MathJax>
          </div>
          <p>
            where <MathJax inline>{"$\\hiddStatet \\in \\Q^\\hiddDim$"}</MathJax> is the hidden state vector at time step <MathJax inline>{"$t$"}</MathJax>, <MathJax inline>{"$\\rnnInitstate \\in \\Q^\\hiddDim$"}</MathJax> is an initialization parameter, <MathJax inline>{"$\\symt \\in \\alphabet$"}</MathJax> is the input symbol, <MathJax inline>{"$\\inEmbedding: \\alphabet \\to \\Q^\\embedDim$"}</MathJax> is a symbol representation function, <MathJax inline>{"$\\recMtx \\in \\Q^{\\hiddDim \\times \\hiddDim}$"}</MathJax> and <MathJax inline>{"$\\inMtx \\in \\Q^{\\hiddDim \\times \\embedDim}$"}</MathJax> are parameter matrices, <MathJax inline>{"$\\biasVec \\in \\Q^{\\hiddDim}$"}</MathJax> is a bias vector, and <MathJax inline>{"$\\activation: \\Q^\\hiddDim \\to \\Q^\\hiddDim$"}</MathJax> is an element-wise non-linear activation function.
          </p>
          <p>
            Because <MathJax inline>{"$\\hiddStatet$"}</MathJax> encodes the string consumed by the Elman RNN, we also use the notation <MathJax inline>{"$\\hiddState(\\str)$"}</MathJax> to denote the result of applying the recurrence over the string <MathJax inline>{"$\\str = \\sym_1 \\cdots \\sym_t$"}</MathJax>.
          </p>
          <p>
            The choice of non-linear activation <MathJax inline>{"$\\activation$"}</MathJax> is crucial for the representational capacity of the Elman RNN. Common examples include <MathJax inline>{"$\\ReLU(x) = \\max(0, x)$"}</MathJax>, the Heaviside function <MathJax inline>{"$\\heaviside(x) = \\ind{x > 0}$"}</MathJax>, and the sigmoid <MathJax inline>{"$\\sigmoid(x) = \\frac{1}{1 + \\exp(-x)}$"}</MathJax>. <MathJax inline>{"$\\ReLU$"}</MathJax> is the most common in modern deep learning and the focus of our analysis.
          </p>
          <p>
            To define an autoregressive language model, an Elman RNN constructs a distribution over the next symbol in <MathJax inline>{"$\\eosalphabet$"}</MathJax> by transforming the hidden state using a function <MathJax inline>{"$\\mlp: \\R^\\hiddDim \\to \\R^\\eosnsymbols$"}</MathJax>.
          </p>
          <h4 className="mt-6 mb-2 font-semibold">Definition</h4>
          <div className="flex justify-center my-4">
            <MathJax>{`
              $$
              \\text{Elman LM:} \\\\
              \\text{Let } \\rnn \\text{ be an Elman RNN and } \\mlp: \\R^{\\hiddDim} \\to \\R^{\\eosnsymbols} \\text{ differentiable.} \\\\
              \\text{Conditional distributions:} \\\\
              \\plmRCFun{\\eossym_t}{\\strlt} \\defeq \\projfuncEosalphabetminusFunc{\\mlpFun{\\hiddStatetminus}}_{\\eossym_t}
              $$
            `}</MathJax>
          </div>
          <p>
            The most common choice for the projection function <MathJax inline>{"$\\projfuncEosalphabetminus$"}</MathJax> is the <strong>softmax</strong>:
          </p>
          <div className="flex justify-center my-4">
            <MathJax>{`
              $$
              \\softmaxfunc{\\vx}{n} = \\frac{\\exp \\left(\\invtemp \, \\evx_n\\right)}{\\sum_{n' = 1}^{\\setsize} \\exp{\\left(\\invtemp \, \\evx_{n'}\\right)}}
              $$
            `}</MathJax>
          </div>
          <p>
            where <MathJax inline>{"$\\invtemp \\in \\R$"}</MathJax> is the <strong>inverse temperature</strong> parameter. Softmax may be viewed as the solution to the following convex optimization problem:
          </p>
          <div className="flex justify-center my-4">
            <MathJax>{`
              $$
              \\softmaxfunc{\\vx}{} = \\argmax_{\\vz\\in\\Simplexnminus} \\vz^\\top\\vx+\\entropyFun{\\vz}
              $$
            `}</MathJax>
          </div>
          <p>
            where <MathJax inline>{"$\\entropyFun{\\vz} = -\\sum_{n=1}^\\setsize z_n\\log z_n$"}</MathJax> is the Shannon entropy.
          </p>
          <p>
            An important limitation of the softmax is that it implies the LM has full support, i.e., an Elman LM with a softmax projection assigns positive probability to all strings in <MathJax inline>{"$\\kleene{\\alphabet}$"}</MathJax>.
          </p>
          <p>
            To construct Elman LMs without full support, we also consider the <strong>sparsemax</strong> as our projection function:
          </p>
          <div className="flex justify-center my-4">
            <MathJax>{`
              $$
              \\sparsemaxfunc{\\vx}{} = \\argmin_{\\vz\\in \\Simplexnminus} \\norm{\\vz - \\vx}^2_2
              $$
            `}</MathJax>
          </div>
          <p>
            In contrast to softmax, sparsemax is the <strong>identity</strong> on <MathJax inline>{"$\\Simplexnminus$"}</MathJax>, i.e., <MathJax inline>{"$\\sparsemaxfunc{\\vx}{} = \\vx$"}</MathJax> for <MathJax inline>{"$\\vx \\in \\Simplexnminus$"}</MathJax>.
          </p>
          <h4 className="mt-6 mb-2 font-semibold">Bounded Precision</h4>
          <p>
            We assume the hidden states <MathJax inline>{"$\\hiddStatet$"}</MathJax> to be rational vectors. An important consideration in processing strings <MathJax inline>{"$\\str \\in \\kleene{\\alphabet}$"}</MathJax> is the number of bits required to represent the entries in <MathJax inline>{"$\\hiddStatet$"}</MathJax> and how this scales with the string length <MathJax inline>{"$|\\str|$"}</MathJax>. This motivates the definition of precision.
          </p>
          <h4 className="mt-6 mb-2 font-semibold">Definition</h4>
          <div className="flex justify-center my-4">
            <MathJax>{`
              $$
              \\precisionFun{\\str} = \\max_{d\\in[\\hiddDim]}\\min_{\\substack{p,q\\in\\N,\\\\ \\frac{p}{q}=\\hiddState\\left(\\str\\right)_{d}}} \\lceil\\log_2 p\\rceil + \\lceil\\log_2 q\\rceil
              $$
            `}</MathJax>
          </div>
          <div>
            <span>We say an Elman RNN is of:</span>
            <ul className="list-disc ml-6">
              <li><strong>Constant precision</strong> if <MathJax inline>{"$\\precisionFun{\\str} = \\mathcal{O}(1)$"}</MathJax></li>
              <li><strong>Logarithmically bounded precision</strong> if <MathJax inline>{"$\\precisionFun{\\str} = \\mathcal{O}(\\log|\\str|)$"}</MathJax></li>
              <li><strong>Linearly bounded precision</strong> if <MathJax inline>{"$\\precisionFun{\\str} = \\mathcal{O}(|\\str|)$"}</MathJax></li>
              <li><strong>Unbounded precision</strong> if <MathJax inline>{"$\\precisionFun{\\str}$"}</MathJax> cannot be bounded by a function of <MathJax inline>{"$|\\str|$"}</MathJax></li>
            </ul>
            <span>The constructions in the literature range from constant to unbounded precision.</span>
          </div>
        </>
      ),
    },
    {
      title: "Representational Capacity of Recurrent Neural Networks",
      content: (
        <>
          <p className="mb-3">
            We present the following connections between RNNs and classical models of computation:
          </p>
          <hr />
          <div className="mt-1 border-b pb-4">
            <h4 className="font-semibold text-lg mb-2 mt-4">
              <a href="#rnns-turing-completeness">Turing Completeness of RNNs</a>
            </h4>
            <p>
              <a href="https://doi.org/10.1145/130385.130432">Siegelmann and Sontag (1992)</a> showed that RNNs with rational weights and unbounded computational time are Turing complete. Their construction shows how the internal state of the Turing machine can be encoded in the hidden state of the RNN.
            </p>
          </div>
          <div className="mt-1 border-b pb-4">
            <h4 className="font-semibold text-lg mb-2 mt-4">
              <a href="#rnns-pfsas">RNNs and Finite-state Automata</a>
            </h4>
            <p>
              This comparison is as old as both computational models: Some studies go back to the 40's and one of the main results we will discuss is Minsky's result from the 50's.
            </p>
          </div>
          <div className="mt-1 border-b pb-4">
            <h4 className="font-semibold text-lg mb-2 mt-4">
              <a href="#rnns-counters">RNNs and Counter Machines</a>
            </h4>
            <p>
              Somewhat restricted RNNs are easy to link to finite-state automata. What happens if we relax these restrictions a bit? It turns out that long short-term memory units (LSTMs) are most naturally interpreted and seen as implementing some counter behavior.
            </p>
          </div>
        </>
      ),
    },
    {
      title: "References",
      content: (
        <ul className="list-disc ml-6">
          <li>Bauwens (2013) Bruno Bauwens. 2013. <a href="URL_HERE">Upper semicomputable sumtests for lower semicomputable semimeasures</a>. <i>arXiv preprint arXiv:1312.1718</i>.</li>
          <li>Buchsbaum et al. (2000) Adam L. Buchsbaum, Raffaele Giancarlo, and Jeffery R. Westbrook. 2000. <a href="URL_HERE">On the determinization of weighted finite automata</a>. <i>SIAM Journal on Computing</i>, 30(5):1502–1531.</li>
          <li>Burkhard and Varaiya (1971) W.A. Burkhard and P.P. Varaiya. 1971. <a href="URL_HERE">Complexity problems in real time languages</a>. <i>Information Sciences</i>, 3(1):87–100.</li>
          <li>Chen et al. (2018) Yining Chen, Sorcha Gilroy, Andreas Maletti, Jonathan May, and Kevin Knight. 2018. <a href="URL_HERE">Recurrent neural networks as weighted language recognizers</a>. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 2261–2271, New Orleans, Louisiana. Association for Computational Linguistics.</li>
          <li>Chung and Siegelmann (2021) Stephen Chung and Hava Siegelmann. 2021. <a href="URL_HERE">Turing completeness of bounded-precision recurrent neural networks</a>. In Advances in Neural Information Processing Systems, volume 34, pages 28431–28441. Curran Associates, Inc.</li>
          <li>Deletang et al. (2023) Gregoire Deletang, Anian Ruoss, Jordi Grau-Moya, Tim Genewein, Li Kevin Wenliang, Elliot Catt, Chris Cundy, Marcus Hutter, Shane Legg, Joel Veness, and Pedro A Ortega. 2023. <a href="URL_HERE">Neural networks and the Chomsky hierarchy</a>. In The Eleventh International Conference on Learning Representations.</li>
          <li>Du et al. (2023) Li Du, Lucas Torroba Hennigen, Tiago Pimentel, Clara Meister, Jason Eisner, and Ryan Cotterell. 2023. <a href="URL_HERE">A measure-theoretic characterization of tight language models</a>. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9744–9770, Toronto, Canada. Association for Computational Linguistics.</li>
          <li>Elman (1990) Jeffrey L. Elman. 1990. <a href="URL_HERE">Finding structure in time</a>. <i>Cognitive Science</i>, 14(2):179–211.</li>
          <li>Forst and Hoffmann (2010) Wilhelm Forst and Dieter Hoffmann. 2010. <a href="URL_HERE">Optimization—Theory and Practice</a>. Springer New York, NY.</li>
          <li>Gill (1974) John T. Gill. 1974. <a href="URL_HERE">Computational complexity of probabilistic turing machines</a>. In Proceedings of the Sixth Annual ACM Symposium on Theory of Computing, STOC ’74, page 91–95, New York, NY, USA. Association for Computing Machinery.</li>
          <li>Hao et al. (2018) Yiding Hao, William Merrill, Dana Angluin, Robert Frank, Noah Amsel, Andrew Benz, and Simon Mendelsohn. 2018. <a href="URL_HERE">Context-free transductions with neural stacks</a>. In Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, pages 306–315, Brussels, Belgium. Association for Computational Linguistics.</li>
          <li>Harrison and Havel (1972) Michael A. Harrison and Ivan M. Havel. 1972. <a href="URL_HERE">Real-time strict deterministic languages</a>. <i>SIAM Journal on Computing</i>, 1(4):333–349.</li>
          <li>Hewitt et al. (2020) John Hewitt, Michael Hahn, Surya Ganguli, Percy Liang, and Christopher D. Manning. 2020. <a href="URL_HERE">RNNs can generate bounded hierarchical languages with optimal memory</a>. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1978–2010, Online. Association for Computational Linguistics.</li>
          <li>Hochreiter and Schmidhuber (1997) Sepp Hochreiter and Jürgen Schmidhuber. 1997. <a href="URL_HERE">Long short-term memory</a>. <i>Neural Computation</i>, 9(8):1735–1780.</li>
          <li>Hopcroft et al. (2001) John E. Hopcroft, Rajeev Motwani, and Jeffrey D. Ullman. 2001. <a href="URL_HERE">Introduction to Automata Theory, Languages, and Computation, 3 edition</a>. Pearson.</li>
          <li>Icard (2020) Thomas F. Icard. 2020. <a href="URL_HERE">Calibrating generative models: The probabilistic Chomsky–Schützenberger hierarchy</a>. <i>Journal of Mathematical Psychology</i>, 95:102308.</li>
          <li>Kleene (1956) S. C. Kleene. 1956. <a href="URL_HERE">Representation of Events in Nerve Nets and Finite Automata</a>, pages 3–42. Princeton University Press, Princeton.</li>
          <li>Knuth and Yao (1976) Donald Ervin Knuth and Andrew Chi-Chih Yao. 1976. <a href="URL_HERE">The complexity of nonuniform random number generation</a>. In Algorithms and Complexity: New Directions and Recent Results, page 357–428, USA. Academic Press, Inc.</li>
          <li>Korsky and Berwick (2019) Samuel A. Korsky and Robert C. Berwick. 2019. <a href="URL_HERE">On the computational power of RNNs</a>. <i>arXiv preprint arXiv:1906.06349</i>.</li>
          <li>Li and Vitányi (2008) Ming Li and Paul M.B. Vitányi. 2008. <a href="URL_HERE">An Introduction to Kolmogorov Complexity and Its Applications, 3 edition</a>. Springer Publishing Company, Incorporated.</li>
          <li>Linzen et al. (2016) Tal Linzen, Emmanuel Dupoux, and Yoav Goldberg. 2016. <a href="URL_HERE">Assessing the ability of LSTMs to learn syntax-sensitive dependencies</a>. <i>Transactions of the Association for Computational Linguistics</i>, 4:521–535.</li>
          <li>Martins and Astudillo (2016) André F. T. Martins and Ramón F. Astudillo. 2016. <a href="URL_HERE">From softmax to sparsemax: A sparse model of attention and multi-label classification</a>. In Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48, ICML’16, page 1614–1623.</li>
          <li>Meister et al. (2023) Clara Meister, Tiago Pimentel, Gian Wiher, and Ryan Cotterell. 2023. <a href="URL_HERE">Locally typical sampling</a>. <i>Transactions of the Association for Computational Linguistics</i>, 11:102–121.</li>
          <li>Merrill (2019) William Merrill. 2019. <a href="URL_HERE">Sequential neural networks as automata</a>. In Proceedings of the Workshop on Deep Learning and Formal Languages: Building Bridges, pages 1–13, Florence. Association for Computational Linguistics.</li>
          <li>Merrill et al. (2020) William Merrill, Gail Weiss, Yoav Goldberg, Roy Schwartz, Noah A. Smith, and Eran Yahav. 2020. <a href="URL_HERE">A formal hierarchy of RNN architectures</a>. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 443–459, Online. Association for Computational Linguistics.</li>
          <li>Minsky (1967) Marvin L. Minsky. 1967. <a href="URL_HERE">Computation: Finite and Infinite Machines</a>. Prentice-Hall, Inc., USA.</li>
          <li>Minsky (1954) Marvin Lee Minsky. 1954. <a href="URL_HERE">Neural Nets and the Brain Model Problem</a></li>
          <li>Mohri (1997) Mehryar Mohri. 1997. <a href="URL_HERE">Finite-state transducers in language and speech processing</a>. <i>Computational Linguistics</i>, 23(2):269–311.</li>
          <li>Orvieto et al. (2023) Antonio Orvieto, Samuel L Smith, Albert Gu, Anushan Fernando, Caglar Gulcehre, Razvan Pascanu, and Soham De. 2023. <a href="URL_HERE">Resurrecting recurrent neural networks for long sequences</a>.</li>
          <li>Peng et al. (2023) Bo Peng, Eric Alcaide, Quentin Anthony, Alon Albalak, Samuel Arcadinho, Huanqi Cao, Xin Cheng, Michael Chung, Matteo Grella, Kranthi Kiran GV, Xuzheng He, Haowen Hou, Przemyslaw Kazienko, Jan Kocon, Jiaming Kong, Bartlomiej Koptyra, Hayden Lau, Krishna Sri Ipsit Mantri, Ferdinand Mom, Atsushi Saito, Xiangru Tang, Bolun Wang, Johan S. Wind, Stansilaw Wozniak, Ruichong Zhang, Zhenyuan Zhang, Qihang Zhao, Peng Zhou, Jian Zhu, and Rui-Jie Zhu. 2023. <a href="URL_HERE">RWKV: Reinventing RNNs for the transformer era</a>. <i>arXiv preprint arXiv:2305.13048</i>.</li>
          <li>Pittl and Yehudai (1983) Jan Pittl and Amiram Yehudai. 1983. <a href="URL_HERE">Constructing a realtime deterministic pushdown automaton from a grammar</a>. <i>Theoretical Computer Science</i>, 22(1):57–79.</li>
          <li>Pérez et al. (2019) Jorge Pérez, Javier Marinković, and Pablo Barceló. 2019. <a href="URL_HERE">On the Turing completeness of modern neural network architectures</a>. In International Conference on Learning Representations.</li>
          <li>Qiu et al. (2020) XiPeng Qiu, TianXiang Sun, YiGe Xu, YunFan Shao, Ning Dai, and XuanJing Huang. 2020. <a href="URL_HERE">Pre-trained models for natural language processing: A survey</a>. <i>Science China Technological Sciences</i>, 63(10):1872–1897.</li>
          <li>Rabin (1963) Michael O. Rabin. 1963. <a href="URL_HERE">Real time computation</a>. <i>Israel Journal of Mathematics</i>, 1(4):203–211.</li>
          <li>Rosenberg (1967) Arnold L. Rosenberg. 1967. <a href="URL_HERE">Real-time definable languages</a>. <i>Journal of the ACM</i>, 14(4):645–662.</li>
          <li>Roy (2011) Daniel M. Roy. 2011. <a href="URL_HERE">Computability, Inference and Modeling in Probabilistic Programming</a>. Ph.D. thesis, Massachusetts Institute of Technology, USA. AAI0823858.</li>
          <li>Siegelmann and Sontag (1992) Hava T. Siegelmann and Eduardo D. Sontag. 1992. <a href="URL_HERE">On the computational power of neural nets</a>. In Proceedings of the Fifth Annual Workshop on Computational Learning Theory, COLT ’92, page 440–449, New York, NY, USA. Association for Computing Machinery.</li>
          <li>Sipser (2013) Michael Sipser. 2013. <a href="URL_HERE">Introduction to the Theory of Computation, 3 edition</a>. Cengage Learning.</li>
          <li>Svete and Cotterell (2023) Anej Svete and Ryan Cotterell. 2023. <a href="URL_HERE">Recurrent neural language models as probabilistic finite-state automata</a>. <i>arXiv preprint arXiv:2310.05161</i>.</li>
          <li>Tadaki et al. (2010) Kohtaro Tadaki, Tomoyuki Yamakami, and Jack C.H. Lin. 2010. <a href="URL_HERE">Theory of one-tape linear-time Turing machines</a>. <i>Theoretical Computer Science</i>, 411(1):22–43.</li>
          <li>Vaswani et al. (2017) Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. <a href="URL_HERE">Attention is all you need</a>. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.</li>
          <li>Weihrauch (2000) Klaus Weihrauch. 2000. <a href="URL_HERE">Computable Analysis - An Introduction</a>. Texts in Theoretical Computer Science. An EATCS Series. Springer.</li>
          <li>Weiss et al. (2018) Gail Weiss, Yoav Goldberg, and Eran Yahav. 2018. <a href="URL_HERE">On the practical computational power of finite precision RNNs for language recognition</a>. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 740–745, Melbourne, Australia. Association for Computational Linguistics.</li>
          <li>Welleck et al. (2020) Sean Welleck, Ilia Kulikov, Jaedeok Kim, Richard Yuanzhe Pang, and Kyunghyun Cho. 2020. <a href="URL_HERE">Consistency of a recurrent language model with respect to incomplete decoding</a>. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 5553–5568, Online. Association for Computational Linguistics.</li>
          <li>Zhou et al. (2023) Wangchunshu Zhou, Yuchen Eleanor Jiang, Peng Cui, Tiannan Wang, Zhenxin Xiao, Yifan Hou, Ryan Cotterell, and Mrinmaya Sachan. 2023. <a href="URL_HERE">RecurrentGPT: Interactive generation of (arbitrarily) long text</a>. <i>arXiv preprint arXiv:2305.13304</i>.</li>
        </ul>
      ),
    },
  ];

  return <ContentLayout title="Recurrent Neural Networks and Language Models" intro={intro} sections={sections} />;
}