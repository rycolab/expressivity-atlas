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
              \text{An \textbf{Elman RNN} } \\rnn = \elmanrnntuple \text{ is an RNN with the following hidden state recurrence:}
              \\
              \begin{align}
                \hiddStatetzero &= \rnnInitstate  \quad\quad\quad\quad\quad\quad\quad\quad\quad\quad\,\,\,{(t=0)} \\
                \hiddStatet &= \activation\left(\recMtx \vhtminus + \inMtx \inEmbedding\left(\symt\right) + \biasVec \right) \,\, {(t>0)}
              \end{align}
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
              \text{Let } \rnn \text{ be an Elman RNN and } \mlp: \R^\hiddDim \to \R^\eosnsymbols \text{ a differentiable function.}
              \\
              \text{An \textbf{Elman LM} is an LM whose conditional distributions are defined by projecting } \mlpFun{\hiddStatetminus} \text{ onto the probability simplex } \SimplexEosalphabetminus:
              \\
              \plmRCFun{\eossym_t}{\strlt} \defeq \projfuncEosalphabetminusFunc{\mlpFun{\hiddStatetminus}}_{\eossym_t}.
            `}</MathJax>
          </div>
          <p>
            The most common choice for the projection function <MathJax inline>{"$\\projfuncEosalphabetminus$"}</MathJax> is the <strong>softmax</strong>:
          </p>
          <div className="flex justify-center my-4">
            <MathJax>{`
              \softmaxfunc{\vx}{n} = \frac{\exp \left(\invtemp \, \evx_n\right)}{\sum_{n' = 1}^{\setsize} \exp{\left(\invtemp \, \evx_{n'}\right)}}
            `}</MathJax>
          </div>
          <p>
            where <MathJax inline>{"$\\invtemp \\in \\R$"}</MathJax> is the <strong>inverse temperature</strong> parameter. Softmax may be viewed as the solution to the following convex optimization problem:
          </p>
          <div className="flex justify-center my-4">
            <MathJax>{`
              \softmaxfunc{\vx}{} = \argmax_{\vz\in\Simplexnminus} \vz^\top\vx+\entropyFun{\vz}
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
              \sparsemaxfunc{\vx}{} = \argmin_{\vz\in \Simplexnminus} \norm{\vz - \vx}^2_2
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
              \precisionFun{\str} = \max_{d\in[\hiddDim]}\min_{\substack{p,q\in\N,\\ \frac{p}{q}=\hiddState\left(\str\right)_{d}}} \lceil\log_2 p\rceil + \lceil\log_2 q\rceil
            `}</MathJax>
          </div>
          <p>
            We say an Elman RNN is of:
            <ul className="list-disc ml-6">
              <li><strong>Constant precision</strong> if <MathJax inline>{"$\\precisionFun{\\str} = \\mathcal{O}(1)$"}</MathJax></li>
              <li><strong>Logarithmically bounded precision</strong> if <MathJax inline>{"$\\precisionFun{\\str} = \\mathcal{O}(\\log|\\str|)$"}</MathJax></li>
              <li><strong>Linearly bounded precision</strong> if <MathJax inline>{"$\\precisionFun{\\str} = \\mathcal{O}(|\\str|)$"}</MathJax></li>
              <li><strong>Unbounded precision</strong> if <MathJax inline>{"$\\precisionFun{\\str}$"}</MathJax> cannot be bounded by a function of <MathJax inline>{"$|\\str|$"}</MathJax></li>
            </ul>
            The constructions in the literature range from constant to unbounded precision.
          </p>
        </>
      ),
    },
    {
      title: "Representational Capacity of Recurrent Neural Networks",
      content: (
        <>
          <p>
            We present the following connections between RNNs and classical models of computation:
          </p>
          <div className="border-b pb-4 mb-4">
            <h4 className="font-bold text-lg mt-4 mb-2">
              <a href="#rnns-turing-completeness">Turing Completeness of RNNs</a>
            </h4>
            <p>
              <a href="https://doi.org/10.1145/130385.130432">Siegelmann and Sontag (1992)</a> showed that RNNs with rational weights and unbounded computational time are Turing complete. Their construction shows how the internal state of the Turing machine can be encoded in the hidden state of the RNN.
            </p>
          </div>
          <div className="border-b pb-4 mb-4">
            <h4 className="font-bold text-lg mt-4 mb-2">
              <a href="#rnns-pfsas">RNNs and Finite-state Automata</a>
            </h4>
            <p>
              This comparison is as old as both computational models: Some studies go back to the 40's and one of the main results we will discuss is Minsky's result from the 50's.
            </p>
          </div>
          <div className="border-b pb-4 mb-4">
            <h4 className="font-bold text-lg mt-4 mb-2">
              <a href="#rnns-counters">RNNs and Counter Machines</a>
            </h4>
            <p>
              Somewhat restricted RNNs are easy to link to finite-state automata. If we relax these restrictions, LSTMs are most naturally interpreted as implementing some counter behavior.
            </p>
          </div>
        </>
      ),
    },
  ];

  return <ContentLayout title="Recurrent Neural Networks and Language Models" intro={intro} sections={sections} />;
}