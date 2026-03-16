'use client';

import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import {
  fetchLanguageConfig,
  updateLanguageConfig,
  type SupportedLanguage,
} from '@/lib/api/config';
import { locales, defaultLocale, localeNames, type Locale } from '@/i18n/config';
import { detectPreferredLocale, getHtmlLang } from '@/lib/i18n/locale';

const CONTENT_STORAGE_KEY = 'resume_matcher_content_language';
const UI_STORAGE_KEY = 'resume_matcher_ui_language';

interface LanguageContextValue {
  contentLanguage: SupportedLanguage;
  uiLanguage: Locale;
  isLoading: boolean;
  setContentLanguage: (lang: SupportedLanguage) => Promise<void>;
  setUiLanguage: (lang: Locale) => Promise<void>;
  languageNames: typeof localeNames;
  supportedLanguages: readonly Locale[];
}

const LanguageContext = createContext<LanguageContextValue | undefined>(undefined);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [contentLanguage, setContentLanguageState] = useState<SupportedLanguage>(defaultLocale);
  const [uiLanguage, setUiLanguageState] = useState<Locale>(defaultLocale);
  const [isLoading, setIsLoading] = useState(true);

  // Load cached language preferences first, then reconcile with persisted settings.
  useEffect(() => {
    const loadLanguages = async () => {
      try {
        const detectedLocale = detectPreferredLocale(navigator.languages);

        // Load UI language from localStorage (client-side only)
        const cachedUiLang = localStorage.getItem(UI_STORAGE_KEY);
        const hasCachedUiLanguage = Boolean(cachedUiLang && locales.includes(cachedUiLang as Locale));
        if (hasCachedUiLanguage) {
          setUiLanguageState(cachedUiLang as Locale);
        } else {
          setUiLanguageState(detectedLocale);
          localStorage.setItem(UI_STORAGE_KEY, detectedLocale);
        }

        // Try localStorage first for content language
        const cachedContentLang = localStorage.getItem(CONTENT_STORAGE_KEY);
        if (cachedContentLang && locales.includes(cachedContentLang as Locale)) {
          setContentLanguageState(cachedContentLang as SupportedLanguage);
        }

        // Then fetch persisted language settings from backend to ensure sync
        const config = await fetchLanguageConfig();
        if (config.ui_language && locales.includes(config.ui_language as Locale)) {
          const shouldApplyPersistedUiLanguage =
            hasCachedUiLanguage ||
            config.ui_language !== defaultLocale ||
            detectedLocale === defaultLocale;

          if (shouldApplyPersistedUiLanguage) {
            setUiLanguageState(config.ui_language);
            localStorage.setItem(UI_STORAGE_KEY, config.ui_language);
          }
        }

        if (config.content_language && locales.includes(config.content_language as Locale)) {
          setContentLanguageState(config.content_language);
          localStorage.setItem(CONTENT_STORAGE_KEY, config.content_language);
        }
      } catch (error) {
        console.error('Failed to load language config:', error);
        // Keep using cached/default values
      } finally {
        setIsLoading(false);
      }
    };

    loadLanguages();
  }, []);

  useEffect(() => {
    document.documentElement.lang = getHtmlLang(uiLanguage);
  }, [uiLanguage]);

  const setContentLanguage = useCallback(
    async (lang: SupportedLanguage) => {
      if (!locales.includes(lang as Locale)) {
        console.error(`Unsupported language: ${lang}`);
        return;
      }

      const previousLang = contentLanguage;
      try {
        // Optimistically update UI
        setContentLanguageState(lang);
        localStorage.setItem(CONTENT_STORAGE_KEY, lang);

        // Persist to backend
        await updateLanguageConfig({ content_language: lang });
      } catch (error) {
        console.error('Failed to update content language:', error);
        // Revert on error
        setContentLanguageState(previousLang);
        localStorage.setItem(CONTENT_STORAGE_KEY, previousLang);
      }
    },
    [contentLanguage]
  );

  const setUiLanguage = useCallback(
    async (lang: Locale) => {
      if (!locales.includes(lang)) {
        console.error(`Unsupported UI language: ${lang}`);
        return;
      }

      const previousLang = uiLanguage;
      try {
        setUiLanguageState(lang);
        localStorage.setItem(UI_STORAGE_KEY, lang);
        await updateLanguageConfig({ ui_language: lang });
      } catch (error) {
        console.error('Failed to update UI language:', error);
        setUiLanguageState(previousLang);
        localStorage.setItem(UI_STORAGE_KEY, previousLang);
      }
    },
    [uiLanguage]
  );

  return (
    <LanguageContext.Provider
      value={{
        contentLanguage,
        uiLanguage,
        isLoading,
        setContentLanguage,
        setUiLanguage,
        languageNames: localeNames,
        supportedLanguages: locales,
      }}
    >
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
}
