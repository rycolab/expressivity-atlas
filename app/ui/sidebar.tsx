'use client';

import {
  HomeIcon,
  DocumentIcon,
  FolderIcon,
} from '@heroicons/react/24/outline';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import clsx from 'clsx';
import React from "react";

// Map of links to display in the side navigation.
// Depending on the size of the application, this would be stored in a database.
const links = [
  { name: 'Home', href: '/', icon: HomeIcon },
  {
    name: 'Neural Networks',
    href: '/networks',
    subpages: [
      { name: 'Recurrent Neural Networks', href: '/networks/rnn' },
      { name: 'Transformers', href: '/networks/transformer' },
      { name: 'State-Space Models', href: '/networks/ssm' },
    ],
  },
  {
    name: 'Formal Languages',
    href: '/languages',
  },
  {
    name: 'Formal Logics',
    href: '/logics',
  },
  {
    name: 'Finite Automata',
    href: '/automata',
  },
  {
    name: 'Finite Monoids',
    href: '/monoids',
  },
  {
    name: 'Circuit Complexity',
    href: '/circuits',
  },
  {
    name: 'Programming Languages',
    href: '/programs',
  },
];

export default function SideBar() {
  const pathname = usePathname();
  return (
    <aside className="w-64 bg-gray-50 h-screen flex flex-col">
      <ul>
      {links.map((link) => (
        <li key={link.name}>
          <Link
            href={link.href}
            className={clsx(
              "flex h-[48px] grow items-center justify-center gap-2 rounded-md bg-gray-50 p-3 text-sm font-medium hover:bg-sky-100 hover:text-blue-600 md:flex-none md:justify-start md:p-2 md:px-3",
              {
                "bg-sky-100 text-blue-600": pathname === link.href,
              }
            )}
          >
            <FolderIcon className="w-6" />
            <p className="hidden md:block">{link.name}</p>
          </Link>
          {pathname.startsWith(link.href) && link.subpages && (
            <ul className="pl-2">
              {link.subpages.map((subpage) => (
                <li key={subpage.name}>
                  <Link
                    href={subpage.href}
                    className={clsx(
                      "flex h-[40px] items-center justify-start gap-2 rounded-md bg-gray-50 p-3 text-sm font-medium hover:bg-sky-100 hover:text-blue-600 pl-20 md:flex-none md:justify-start md:p-2 md:px-3",
                      {
                        "bg-sky-100 text-blue-600": pathname === subpage.href,
                      }
                    )}
                  >
                    <DocumentIcon className="w-5" />
                    <p className="hidden md:block">{subpage.name}</p>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </li>
      ))}
      </ul>
    </aside>
  );
}
