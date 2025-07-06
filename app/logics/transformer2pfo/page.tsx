'use client';

import React from 'react';
import ContentLayout from '@/app/ui/content-layout';
import TransformerDiagram from '@/app/ui/transformer-diagram';
import { transformer1Data } from '@/app/data/transformer1-data';
import { transformer2Data } from '@/app/data/transformer2-data';
import { MathJax } from 'better-react-mathjax';

export default function TransformerPage() {
  return (
    <ContentLayout
      title="From Transformers to Logic"
      intro={
        <p>
          This page explores how transformer models can be interpreted as logical formulas.
          Through simple examples, you'll see how a trained transformer's internal logic can be extracted step by step.
        </p>
      }
      sections={[
        {
          title: 'Example 1',
          content: (
            <div className="flex flex-col items-center gap-5">
              <p>
                We begin with a compact transformer trained to recognize a formal language.
                The model uses just one layer and a single dimension, making its behavior straightforward to interpret.
              </p>
              <p>
                For this example, the transformer is trained on the language&nbsp;
                <MathJax inline>{"$\\kleene{\\alphabet}a\\kleene{\\alphabet}$"}</MathJax>.
              </p>
              <TransformerDiagram {...transformer1Data} />
              <p>
                After training, the transformer captures the logic&nbsp;
                <MathJax inline>{"$\\exists y\\leq x: \\pi_a(y)$"}</MathJax>.
                In other words, it predicts <strong>true</strong> if there is an <code>a</code> at or before position <code>x</code> in the input sequence.
              </p>
              <p>
                We focus on leftmost unique hard attention here for simplicity, but will take a deeper dive into other attention mechanisms <a href="#attentions">later</a>.
              </p>
            </div>
          )
        },
        {
          title: 'Example 2',
          content: (
            <div className="flex flex-col items-center gap-5">
              <p>
                Next, let's examine the same model at an earlier training checkpoint—before it achieves perfect accuracy.
              </p>
              <p>
                We can extract logical expressions from the model's parameters to understand what it has learned so far.
              </p>
              <TransformerDiagram {...transformer2Data} />
              <p>
                Here, the model's output corresponds to&nbsp;
                <MathJax inline>{"$\\pi_a(x)$"}</MathJax>:
                it predicts <strong>true</strong> only if the <strong>current token</strong> is <code>a</code>.
              </p>
              <p>
                One might assume that the attention mechanism has failed to learn the full logic&nbsp;
                <MathJax inline>{"$\\exists y\\leq x: \\pi_a(y)$"}</MathJax>.
                However, a closer look reveals that attention already linearly separates positive and negative cases in its representations.
                The real bottleneck is the feedforward sublayer, which has not yet learned to map those representations to the correct final output.
              </p>
            </div>
          )
        },
      ]}
    />
  );
}
