'use client';

import React from 'react';
import Link from 'next/link';
import { localeFlags } from '@/i18n/config';
import { useLanguage } from '@/lib/context/language-context';
import { useTranslations } from '@/lib/i18n';

export default function Hero() {
  const { t } = useTranslations();
  const { uiLanguage, setUiLanguage, languageNames, supportedLanguages } = useLanguage();

  const buttonClass =
    'group relative border border-black bg-transparent px-8 py-3 font-mono text-sm font-bold uppercase text-blue-700 transition-all duration-200 ease-in-out hover:bg-blue-700 hover:text-[#F0F0E8] hover:-translate-y-1 hover:-translate-x-1 hover:shadow-[4px_4px_0px_0px_#000000] active:translate-x-0 active:translate-y-0 active:shadow-none cursor-pointer';
  const languageButtonClass =
    'border border-black px-3 py-2 font-mono text-xs font-bold uppercase tracking-wide transition-all duration-150 ease-out';

  return (
    <section
      className="h-screen w-full p-4 md:p-12 lg:p-24 bg-[#F0F0E8]"
      style={{
        backgroundImage:
          'linear-gradient(rgba(29, 78, 216, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(29, 78, 216, 0.1) 1px, transparent 1px)',
        backgroundSize: '40px 40px',
      }}
    >
      <div className="flex h-full w-full flex-col items-center justify-center border border-black text-blue-700 bg-[#F0F0E8] shadow-[12px_12px_0px_0px_rgba(0,0,0,0.1)]">
        <div className="mb-10 flex w-full max-w-5xl flex-col gap-3 px-6 pt-6 md:px-10 md:pt-10">
          <p className="font-mono text-xs font-bold uppercase tracking-[0.3em] text-black">
            {t('settings.uiLanguage')}
          </p>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
            {supportedLanguages.map((lang) => (
              <button
                key={lang}
                type="button"
                onClick={() => void setUiLanguage(lang)}
                className={`${languageButtonClass} ${
                  uiLanguage === lang
                    ? 'bg-blue-700 text-white shadow-[4px_4px_0px_0px_#000000]'
                    : 'bg-[#F0F0E8] text-black hover:-translate-y-1 hover:-translate-x-1 hover:shadow-[4px_4px_0px_0px_#000000]'
                }`}
              >
                {localeFlags[lang]} {languageNames[lang]}
              </button>
            ))}
          </div>
        </div>

        <h1 className="mb-12 text-center font-mono text-6xl font-bold uppercase leading-none tracking-tighter md:text-8xl lg:text-9xl selection:bg-blue-700 selection:text-white">
          {t('home.brandLine1')}
          <br />
          {t('home.brandLine2')}
        </h1>

        <div className="flex flex-col gap-4 md:flex-row md:gap-12">
          <a
            href="https://github.com/srbhr/Resume-Matcher"
            target="_blank"
            rel="noopener noreferrer"
            className={buttonClass}
          >
            GitHub
          </a>
          <a
            href="https://resumematcher.fyi"
            target="_blank"
            rel="noopener noreferrer"
            className={buttonClass}
          >
            {t('home.docs')}
          </a>
          <Link href="/dashboard" className={buttonClass}>
            {t('home.launchApp')}
          </Link>
        </div>
      </div>
    </section>
  );
}
