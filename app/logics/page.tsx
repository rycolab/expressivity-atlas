'use client';

import React from 'react';
import ContentLayout from '@/app/ui/content-layout'; 
import TransformerDiagram from '@/app/ui/transformer-diagram'; 

export default function TransformerPage() {
  return (
    <ContentLayout
      title="From Transformers to Logic"
      intro={
        <>
          This page demonstrates how to translate a transformer with arbitrary weights into its equivalent logical formulas. As an example, we use a simplified transformer recognizer with a single dimension and one layer. The weights are trained on a synthetic formal language recognition task.
        </>
      }
      sections={[
        {
          title: 'Example 1',
          content: <TransformerDiagram />,
        },
        {
          title: 'Example 2',
          content: <TransformerDiagram />,
        },
      ]}
    />
  );
}
