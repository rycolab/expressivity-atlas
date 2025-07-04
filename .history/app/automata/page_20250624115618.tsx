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
            An <strong>alphabet</strong> <MathJax inline>{"$\\alphabet$"}</MathJax> is a finite, non-empty set of symbols. The <strong>Kleene closure</strong> of an alphabet <MathJax inline>{"$\\alphabet$"}</MathJax> is the set of all finite strings over <MathJax inline>{"$\\alphabet$"}</MathJax> and is denoted by <MathJax inline>{"$\\kleene{\\alphabet}$"}</MathJax>:
          </p>
          <MathJax>{"$\\kleene{\\alphabet} \\defeq \\bigcup_{n=0}^\\infty \\alphabet^n$"}</MathJax>
          <p>
            where <MathJax inline>{"$\\alphabet^n$"}</MathJax> is the set of all strings of length <MathJax inline>{"$n$"}</MathJax> over <MathJax inline>{"$\\alphabet$"}</MathJax>.
            A <strong>language model</strong> (LM) is a probability distribution over <MathJax inline>{"$\\kleene{\\alphabet}$"}</MathJax>, i.e., the strings of symbols from a given alphabet. We denote a language model by <MathJax inline>{"$\\plm$"}</MathJax>.
          </p>
          <h4 className="mt-6 mb-2 font-semibold">Autoregressive Language Models</h4>
          <p>
            Most modern neural language models define the probability <MathJax inline>{"$\\plm(\\str)$"}</MathJax> of a string <MathJax inline>{"$\\str \\in \\kleene{\\alphabet}$"}</MathJax> <strong>autoregressively</strong> as a product of conditional probability distributions:
          </p>
          <MathJax>{"$\\plm(\\str) \\defeq \\plm(\\eos|\\str) \\prod_{t=1}^{|\\str|} \\plm(\\symt|\\strlt)$"}</MathJax>
          <p>
            where <MathJax inline>{"$\\eos$"}</MathJax> is a special end-of-sequence symbol not in <MathJax inline>{"$\\alphabet$"}</MathJax>. A language model expressed this way is called <strong>autoregressive</strong>.
          </p>
          <p>
            Let <MathJax inline>{"$\\eosalphabet \\defeq \\alphabet \\cup \\{\\eos\\}$"}</MathJax>. Each <MathJax inline>{"$\\plm(\\eossym_t|\\strlt)$"}</MathJax> is a distribution over <MathJax inline>{"$\\eosalphabet$"}</MathJax>. We may also consider <MathJax inline>{"$\\varepsilon$"}</MathJax>-augmented language models, where each <MathJax inline>{"$\\plm$"}</MathJax> is a distribution over <MathJax inline>{"$\\eosalphabet \\cup \\{\\varepsilon\\}$"}</MathJax>, with <MathJax inline>{"$\\varepsilon$"}</MathJax> a special symbol for the empty string. This allows the model to perform computations longer than the string it generates. An autoregressive language model is called <strong>real-time</strong> if each <MathJax inline>{"$\\plm$"}</MathJax> is only a distribution over <MathJax inline>{"$\\eosalphabet$"}</MathJax>.
          </p>
          <h4 className="mt-6 mb-2 font-semibold">Equivalence</h4>
          <p>
            To study the relationship between classes of language models, we need a notion of equivalence. We use <strong>weak equivalence</strong>:
          </p>
          <p>
            <strong>Definition:</strong> Two language models <MathJax inline>{"$\\plm$"}</MathJax> and <MathJax inline>{"$\\qlm$"}</MathJax> over <MathJax inline>{"$\\kleene{\\alphabet}$"}</MathJax> are weakly equivalent if <MathJax inline>{"$\\plm(\\str) = \\qlm(\\str)$"}</MathJax> for all <MathJax inline>{"$\\str \\in \\kleene{\\alphabet}$"}</MathJax>.
          </p>
        </>
      ),
    },
    {
      title: "Regular Language Models and Probabilistic Finite-State Automata",
      content: (
        <>
          <h4 className="mt-6 mb-2 font-semibold">Probabilistic Finite-State Automata</h4>
          <p>
            Probabilistic finite-state automata (PFSAs) are a well-understood real-time computational model.
          </p>
          <p>
            <strong>Definition:</strong> A PFSA is a 5-tuple <MathJax inline>{"$\\wfsatuple$"}</MathJax> where <MathJax inline>{"$\\alphabet$"}</MathJax> is an alphabet, <MathJax inline>{"$\\states$"}</MathJax> is a finite set of states, <MathJax inline>{"$\\trans \\subseteq \\states \\times \\alphabet \\times \\Qnonneg \\times \\states$"}</MathJax> is a finite set of weighted transitions, and <MathJax inline>{"$\\initialf, \\finalf: \\states \\to \\Qnonneg$"}</MathJax> assign initial and final weights. For all <MathJax inline>{"$\\stateq \\in \\states$"}</MathJax>:
          </p>
          <MathJax>{"$\\sum_{\\stateq \\in \\states} \\initialf(\\stateq) = 1$"}</MathJax>
          <MathJax>{"$\\sum_{\\edge{\\stateq}{\\sym}{w}{\\stateq'}} w + \\finalf(\\stateq) = 1$"}</MathJax>
          <h4 className="mt-6 mb-2 font-semibold">Basic Concepts</h4>
          <p>
            A PFSA is <strong>deterministic</strong> if it has a unique initial state and, for every state and symbol, at most one outgoing transition with nonzero weight. Any state with <MathJax inline>{"$\\initialf(\\stateq) > 0$"}</MathJax> is <strong>initial</strong>, and if <MathJax inline>{"$\\finalf(\\stateq) > 0$"}</MathJax> it is <strong>final</strong>.
          </p>
          <p>
            A <strong>path</strong> <MathJax inline>{"$\\apath$"}</MathJax> of length <MathJax inline>{"$T$"}</MathJax> is a sequence of transitions. The <strong>yield</strong> of a path is the string of symbols along the path. The <strong>prefix weight</strong> is the product of transition and initial weights, and the <strong>weight</strong> of a path also includes the final weight:
          </p>
          <MathJax>{"$\\prefixweight(\\apath) = \\prod_{t=0}^T w_t$"}</MathJax>
          <MathJax>{"$\\weight(\\apath) = \\prod_{t=0}^{T+1} w_t$"}</MathJax>
          <p>
            with <MathJax inline>{"$w_0 = \\initialf(\\stateq_1)$"}</MathJax> and <MathJax inline>{"$w_{T+1} = \\finalf(\\stateq_{T+1})$"}</MathJax>. The sum of weights of all paths yielding a string <MathJax inline>{"$\\str$"}</MathJax> is the <strong>stringsum</strong>:
          </p>
          <MathJax>{"$\\fsa(\\str) = \\sum_{\\apath \\in \\paths(\\fsa, \\str)} \\weight(\\apath)$"}</MathJax>
          <p>
            The stringsum gives the probability of <MathJax inline>{"$\\str$"}</MathJax>. A state is <strong>accessible</strong> if reachable from an initial state, <strong>co-accessible</strong> if it can reach a final state. An automaton where all states are accessible and co-accessible is <strong>trim</strong>.
          </p>
          <h4 className="mt-6 mb-2 font-semibold">PFSAs as Autoregressive Language Models</h4>
          <p>
            A PFSA <MathJax inline>{"$\\fsa$"}</MathJax> induces a language model <MathJax inline>{"$\\plmA$"}</MathJax> over strings <MathJax inline>{"$\\str \\in \\kleene{\\alphabet}$"}</MathJax>. The weights of all available transitions from a state, together with the final weight, define a probability distribution over the next action (transition or halting). This gives a distribution over <MathJax inline>{"$\\eosalphabet$"}</MathJax>:
          </p>
          <MathJax>{`$\\plmA(\\eossym_t|q) = \\begin{cases} \\sum_{\\edge{q}{\\eossym_t}{w}{q'}} w & \\text{if } \\eossym_t \\in \\alphabet \\ \\ \\finalf(q) & \\text{if } \\eossym_t = \\eos \end{cases}$`}</MathJax>
          <p>
            In a PFSA, the probability of <MathJax inline>{"$\\eossym$"}</MathJax> is conditionally independent of <MathJax inline>{"$\\strlt$"}</MathJax> given the state <MathJax inline>{"$q$"}</MathJax>:
          </p>
          <MathJax>{"$\\plmA(\\eossym_t|q, \\strlt) = \\plmA(\\eossym_t|q)$"}</MathJax>
          <p>
            The autoregressive language model is then:
          </p>
          <MathJax>{`$\\plmA(\\eossym_t|\\strlt) = \\sum_q \\plmA(\\eossym_t|q) \\cdot \\frac{\\plmA(q, \\strlt)}{\\sum_{q'} \\plmA(q', \\strlt)}$`}</MathJax>
          <p>
            where <MathJax inline>{"$\\plmA(q, \\strlt)$"}</MathJax> can be written as:
          </p>
          <MathJax>{"$\\plmA(q, \\strlt) = (\\vec{\\lambda}^\\top \\prod_{s=1}^t \\mT^{(y_s)})_q$"}</MathJax>
          <p>
            where <MathJax inline>{"$\\vec{\\lambda}$"}</MathJax> and <MathJax inline>{"$\\mT^{(y_s)}$"}</MathJax> are the vectorized initial function and symbol-specific transition matrices of the PFSA.
          </p>
          <h4 className="mt-6 mb-2 font-semibold">Regular Language Models</h4>
          <p>
            A language model <MathJax inline>{"$\\plm$"}</MathJax> is <strong>regular</strong> if there exists a PFSA <MathJax inline>{"$\\fsa$"}</MathJax> whose induced language model <MathJax inline>{"$\\plmA$"}</MathJax> is weakly equivalent to <MathJax inline>{"$\\plm$"}</MathJax>.
          </p>
          <h4 className="mt-6 mb-2 font-semibold">PFSAs and FSAs</h4>
          <p>
            Although PFSAs share many properties with unweighted (boolean-weighted) finite-state automata, one important difference relates to determinization. In the unweighted case, deterministic and non-deterministic FSAs are equivalent, but for PFSAs, there exist non-deterministic PFSAs with no deterministic equivalent. Thus, non-deterministic PFSAs are strictly more expressive than deterministic ones.
          </p>
        </>
      ),
    },
  ];

  return <ContentLayout title="Automata and Regular Language Models" intro={intro} sections={sections} />;
}