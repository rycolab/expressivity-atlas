import ClassGraph from '@/app/ui/graph';

export default function Page() {
  return (
    <main className="flex flex-col min-h-screen">
      {/* Header */}
      <section className="p-6 max-w-4xl mx-auto">
        <h1 className="text-5xl font-extrabold text-gray-900 mb-4 text-center leading-tight">
          The Expressivity Atlas
        </h1>
        <p className="text-lg text-gray-700 leading-relaxed text-center">
          Welcome to the <strong className="font-semibold text-gray-900">Expressivity Atlas</strong> —
          a curated resource offering a structured overview of the formal methods used to analyze the
          expressive power of neural architectures in language modeling.
        </p>
      </section>

      {/* Graph */}
      <section className="flex-grow px-6 pb-6">
        <ClassGraph />
      </section>
    </main>
  );
}
