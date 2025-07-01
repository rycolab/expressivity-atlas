"use client";

import { MathJax } from "better-react-mathjax";
import ContentLayout from "@/app/ui/ContentLayout";

export default function Page() {
  const intro = (
    <p>
      This page introduces the connection between language models and basic formal language theory, focusing on probabilistic finite-state automata (PFSAs) and their role in modeling regular languages.
    </p>
  );

  const sections = [
    {
      title: "Language Models",
      content: (
        <>
          <p>
            <MathJax>{`
              An <b>alphabet</b> is a finite, non-empty set of symbols.
              We will usually denote an alphabet by \( \alphabet \) but \(\stackalphabet\) will also be used.
              The <b>Kleene closure</b> of an alphabet \( \alphabet \) is the set of all finite strings over \( \alphabet \) and is denoted by \( \kleene{\alphabet} \):
              \\begin{equation}
              \kleene{\alphabet} \defeq \bigcup_{n=0}^\infty \alphabet^n,
              \\end{equation}
              where \( \alphabet^n \) is the set of all strings of length \( n \) over \( \alphabet \).
              A <b>language model</b> (LM) is a probability distribution over \(\kleene{\alphabet}\), i.e, the strings of symbols from a given alphabet.
              We will denote a language model by \( \plm \).
            `}</MathJax>
          </p>
          <h4 className="mt-6 mb-2 font-semibold">Definition</h4>
          <p className="border-t border-b py-2">
            <MathJax>{`
              Most modern neural language models define the probability \( \plm\left(\str\right) \) of a string \( \str \in \kleene{\alphabet} \) <b>autoregressively</b>; as a product of conditional probability distributions \( \plm \), i.e.,
              \\begin{equation}
              \plm\left(\str\right) \defeq \plm\left(\eos\mid\str\right) \prod_{t = 1}^{|\str|} \plm\left(\symt \mid \strlt\right),
              \\end{equation}
              where \( \eos \notin \alphabet \) is a special <u>e</u>nd-<u>o</u>f-<u>s</u>equence symbol.
              A language model expressed this way is called <b>autoregressive</b>.
            `}</MathJax>
          </p>
          <p>
            <MathJax>{`
              Let \( \eosalphabet \defeq \alphabet \cup \left\{\eos\right\} \).
              Then, each \( \plm\left(\eossym_t \mid \strlt\right) \) is a distribution over \( \eosalphabet \).
              Additionally, we may also consider \( \varepsilon \)-augmented language models where each \( \plm \) is a distribution over \( \eosalphabet \cup \{ \varepsilon \} \) where \( \varepsilon \not\in \alphabet \) is a special symbol not in the alphabet that represents the empty string.
              This allows the language model to perform computations <i>longer</i> than the length of the string it generates.
              An autoregressive language model is called <b>real-time</b> if each \( \plm \) is <i>only</i> a distribution over \( \eosalphabet \), i.e., not over \( \eosalphabet \cup \{ \varepsilon \} \).
            `}</MathJax>
          </p>
          <p>
            <MathJax>{`
              To study the relationship between classes of language models, we need a notion of equivalence between language models. In this paper, we will work with weak equivalence.
            `}</MathJax>
          </p>
          <p>
            <MathJax>{`
              <b>Definition:</b> Two language models \( \plm \) and \( \qlm \) over \( \kleene{\alphabet} \) are <b>weakly equivalent</b> if \( \plm\left(\str\right) = \qlm\left(\str\right) \) for all \( \str \in \kleene{\alphabet} \).
            `}</MathJax>
          </p>
        </>
      ),
    },
    {
      title: "Regular Language Models and Probabilistic Finite-State Automata",
      content: (
        <>
          <h3 className="text-xl font-semibold mt-4 mb-2">Probabilistic Finite-State Automata</h3>
          <p>
            Probabilistic finite-state automata are a well-understood real-time computational model.
          </p>
          <h4 className="mt-6 mb-2 font-semibold">Definition</h4>
          <p className="border-t border-b py-2">
            <MathJax>{`
              A <b>probabilistic finite-state automaton</b> (PFSA) is a 5-tuple \( \wfsatuple \) where \( \alphabet \) is an alphabet, \( \states \) is a finite set of states, \( \trans \subseteq \states \times \alphabet \times \Qnonneg \times \states \) is a finite set of weighted transitions
              where we write transitions \( \left(\stateq, \sym, w, \stateq^\prime\right) \in \trans \) as \( \edge{\stateq}{\sym}{w}{\stateq^\prime} \),
              and \( \initialf, \finalf\colon \states \rightarrow \Qnonneg \) are functions that assign each state its initial and final weight, respectively.
              Moreover, for all states \( \stateq \in \states \), \( \trans, \initialf \) and \( \finalf \) satisfy \( \sum_{\stateq \in \states} \initialf\left(\stateq\right) = 1 \), and \( \sum\limits_{\edge{\stateq}{\sym}{w}{\stateq^\prime} \in \trans} w + \finalf\left(\stateq\right) = 1 \).
            `}</MathJax>
          </p>
          <h3 className="text-xl font-semibold mt-4 mb-2">Basic Concepts</h3>
          <p>
            <MathJax>{`
              We next define some basic concepts.
              A PFSA \( \fsa = \wfsatuple \) is <b>deterministic</b> if \( |\set{\stateq \mid \initialfFun{\stateq} > 0}| = 1 \) and, for every \( \stateq \in \states, \sym \in \alphabet \), there is at most one \( \stateq^\prime \in \states \) such that \( \edge{\stateq}{\sym}{w}{\stateq^\prime} \in \trans \) with \( w > 0 \).
              Any state \( \stateq \) where \( \initialfFun{\stateq}>0 \) is called an <b>initial state</b>, and if \( \finalfFun{\stateq} > 0 \), it is called a <b>final state</b>.
              A <b>path</b> \( \apath \) of length \( \pathlen \) is a sequence of subsequent transitions in \( \fsa \), denoted as
              \\[
              \edge{\stateq_1}{\sym_1}{w_1}{\edge{\stateq_2}{\sym_2}{w_2}{\stateq_3} \cdots \edge{\stateq_{\pathlen}}{\sym_{\pathlen}}{w_{\pathlen}}{\stateq_{\pathlen + 1}}}.
              \\]
              The <b>yield</b> of a path is \( \yield\left(\apath\right)\defeq \sym_1 \cdots \sym_{\pathlen} \).
              The <b>prefix weight</b> \( \prefixweight \) of a path \( \apath \) is the product of the transition and initial weights, whereas the <b>weight</b> of a path additionally has the final weight multiplied in.
              In symbols, this means
            `}</MathJax>
          </p>
          <MathJax>{`
            \[
            \prefixweight(\apath)\defeq \prod_{t = 0}^\pathlen w_t,
            \]
          `}</MathJax>
          <MathJax>{`
            \[
            \weight(\apath)\defeq \prod_{t = 0}^{\pathlen+1} w_t,
            \]
          `}</MathJax>
          <p>
            <MathJax>{`
              with \( w_0 \defeq \initialf(\stateq_1) \) and \( w_{\pathlen+1} \defeq \finalf(\stateq_{\pathlen+1}) \).
              We write \( \paths(\fsa, \str) \) for the set of all paths in \( \fsa \) with yield \( \str \).
              The sum of weights of all paths that yield a certain string \( \str\in\kleene{\alphabet} \) is called the <b>stringsum</b>, given in the notation below:
            `}</MathJax>
          </p>
          <MathJax>{`
            \[
            \fsa \left( \str \right) \defeq \sum_{\apath \in \paths\left( \fsa, \str \right) }  \weight \left( \apath \right).
            \]
          `}</MathJax>
          <p>
            <MathJax>{`
              The stringsum gives the probability of the string \( \str \).
              We say a state \( \stateq \in \states \) is <b>accessible</b> if there exists a path with non-zero weight from an initial state to \( \stateq \).
              A state \( \stateq \in \states \) is <b>co-accessible</b> if there exists a path with non-zero weight from \( \stateq \) to a final state.
              An automaton in which all states are accessible and co-accessible is called <b>trim</b>.
            `}</MathJax>
          </p>
          <h4 className="mt-6 mb-2 font-semibold">PFSAs as Autoregressive Language Models</h4>
          <p>
            <MathJax>{`
              We now show how a PFSA $\fsa$ induces a language model $\plmA$ over strings $\str\in\kleene{\alphabet}$.
              The weights of all available transitions of a PFSA in state $\stateq$, together with the final weight, define a probability distribution over the next action, i.e., taking a transition or halting.
              We can translate this into a distribution over $\eosalphabet$ where $\eos$ corresponds to halting.
              In the following, we use the notation  $\eossym\in\eosalphabet$ and $\sym\in\alphabet$ to distinguish between symbols that could be $\eos$ and those that cannot, respectively.
              Specifically, we can define the probability over $\eosalphabet$ as follows
              \\begin{equation}
                \plmACFun{{\eossym_t}}{{\stateq}, \strlt} = 
                \begin{cases}
                      {\displaystyle \sum_{\edge{{\stateq}}{{\eossym_t}}{w}{\stateq'}}} w  & \ifcond {\eossym_t}\in\alphabet\\
                      \finalf\left({\stateq}\right) & \ifcond {\eossym_t} = \eos.
                  \end{cases}\\label{eq:plmpfsa}
              \\end{equation}
              Moreover, in a PFSA, the probability of $\eossym$ is conditionally independent of $\strlt$ given the state $q$, i.e.,
              \\begin{equation}
              \plmACFun{\eossym_t}{\stateq, \strlt} = \plmACFun{\eossym_t}{\stateq}.\\label{eq:markov}
              \\end{equation}
              Finally, using the law of total probability, we can define an autoregressive language model as
              \\begin{align}\\label{eq:pfsa-autoregressive}
              &\plmACFun{\eossym_t}{\strlt}\defeq \sum_{\stateq \in \states} \plmACFun{\eossym_t}{\stateq}\,\plmACFun{\stateq}{\strlt} \nonumber \\
              &= \sum_{\stateq \in \states} \plmACFun{\eossym_t}{\stateq}\,\frac{\plmA\left(\stateq, \strlt\right)}{\plmA\left(\strlt\right)} \\
              &= \sum_{\stateq \in \states} \plmACFun{\eossym_t}{\stateq}\,\frac{\plmA\left(\stateq, \strlt\right)}{\sum_{\stateq' \in \states} \plmA\left(\stateq', \strlt\right)}, \\label{eq:pfsa-autoregressive-last}
              \\end{align}
              where $\plmA\left(\stateq, \strlt\right)$ can be written as
              \\begin{equation}\\label{eq:next-state}
                  \plmA\left(\stateq, \strlt\right) = \left(\overset{\rightarrow}{\initialf}^{\top} \prod_{s=1}^t \mT^{\left(\sym_s\right)}\right)_\stateq,
              \\end{equation}
              where $\overset{\rightarrow}{\initialf}$ and $\mT^{\left(\sym_s\right)}$ refer to the vectorized initial function and symbol-specific transition matrices of the PFSA $\fsa$, respectively.
            `}</MathJax>
          </p>
          <h4 className="mt-6 mb-2 font-semibold">Definition</h4>
          <p className="border-t border-b py-2">
            <MathJax>{`
              An language model $\plm$ is a <b>regular</b> language model if there exists a PFSA $\fsa$ whose induced language model $\plmA$ is weakly equivalent to $\plmA$.
            `}</MathJax>
          </p>
          <h4 className="mt-6 mb-2 font-semibold">PFSAs and FSAs</h4>
          <p>
            <MathJax>{`
              Although PFSAs share many properties with unweighted (boolean-weighted) finite-state automata, one important difference relates to determinization.
              In the unweighted case, the class of deterministic and non-deterministic FSAs are equivalent, i.e., any non-deterministic FSA has an equivalent deterministic FSA that accepts the same language.
              This result, however, does not hold for PFSAs: There exist PFSAs that admit no deterministic equivalent, meaning that non-deterministic PFSAs are strictly more expressive than deterministic ones.
            `}</MathJax>
          </p>
        </>
      ),
    },
  ];

  return <ContentLayout title="Automata and Regular Language Models" intro={intro} sections={sections} />;
}