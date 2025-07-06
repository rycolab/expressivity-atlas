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
          This page explores how <MathJax inline>{"$\\ptl$"}</MathJax> formulas can be simulated by transformers. 
        </p>
      }
      sections={[
        {
          title: 'Example',
          content: (
            <div className="flex flex-col items-center gap-5">
              <TransformerDiagram {...transformer1Data} />
            </div>
          )
        },
      ]}
    />
  );
}
