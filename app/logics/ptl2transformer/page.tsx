'use client';

import React from 'react';
import ContentLayout from '@/app/ui/content-layout';
import TransformerDiagram2 from '@/app/ui/transformer-diagram2';
import { MathJax } from 'better-react-mathjax';

export default function TransformerPage() {
  return (
    <ContentLayout
      title="From Logic to Transformers"
      intro={
        <p>
          This page explores how <MathJax inline>{"$\\ptl$"}</MathJax> formulas can be simulated by transformers. 
        </p>
      }
      sections={[
        {
          title: 'Example',
          content: (
            <div className="flex flex-col items-center gap-5">
              <p>
                We show how to simulate <MathJax inline>{"$\\past \\pi_a$"}</MathJax> with a transformer under fixed precision.
                We assume the maximum number of positions the the attention can attend to is 1.
              </p>
              <TransformerDiagram2 />
            </div>
          )
        },
      ]}
    />
  );
}
