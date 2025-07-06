'use client';

import { MathJax } from 'better-react-mathjax';
import ContentLayout from '@/app/ui/content-layout';

export default function Page() {
  const intro = (
    <p> 
    </p>
  );

  const sections = [
    {
      title: 'Linear Temporal Logic',
      content: (
        <p>

        </p>
      ),
    },
    {
      title: 'First-Order Logic',
      content: (
        <p>

        </p>
      ),
    },
  ];

  return <ContentLayout title="Formal Logics" intro={intro} sections={sections} />;
}
