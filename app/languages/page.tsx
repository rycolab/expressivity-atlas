'use client';

import { MathJax } from 'better-react-mathjax';
import ContentLayout from '@/app/ui/content-layout';

export default function Page() {
  const intro = (
    <p>
      Formal languages are a foundational concept in theoretical computer science, used to model symbolic structures and computation over strings.
    </p>
  );

  const sections = [
    {
      title: 'Strings and Languages',
      content: (
        <p>
          An <strong>alphabet</strong> <MathJax inline>{"$\\alphabet$"}</MathJax> is a finite, non-empty set of <strong>symbols</strong>. 
          A <strong>string</strong> over <MathJax inline>{"$\\alphabet$"}</MathJax> is a finite sequence of symbols. 
          The <strong>Kleene closure</strong> of <MathJax inline>{"$\\alphabet$"}</MathJax> is <MathJax inline>{"$\\kleene{\\alphabet}$"}</MathJax>.
        </p>
      ),
    },
  ];

  return <ContentLayout title="Formal Languages" intro={intro} sections={sections} />;
}
