import ContentLayout from '@/app/ui/content-layout';
import Link from 'next/link';

export default function Page() {
  const intro = (
    <>
      <p>
        Neural networks form the foundation of modern language modeling. Below are some of the most widely used architectures in this domain.
      </p>
    </>
  );

  const sections = [
    {
      title: 'Network Architectures',
      content: (
        <ul className="list-disc pl-6 space-y-1">
          <li>
            <Link href="/networks/rnn" className="text-blue-600 hover:underline">
              Recurrent Neural Networks
            </Link>
          </li>
          <li>
            <Link href="/networks/transformer" className="text-blue-600 hover:underline">
              Transformer Networks
            </Link>
          </li>
          <li>
            <Link href="/networks/ssm" className="text-blue-600 hover:underline">
              State-Space Models
            </Link>
          </li>
        </ul>
      ),
    },
  ];

  return <ContentLayout title="Neural Networks" intro={intro} sections={sections} />;
}
