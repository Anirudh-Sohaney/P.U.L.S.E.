import { useEffect, type RefObject } from "react";

/** Enhance visible-by-default sections with a single entrance reveal. */
export default function usePublicReveals(rootRef: RefObject<HTMLDivElement>) {
  useEffect(() => {
    const root = rootRef.current;
    if (
      !root ||
      typeof IntersectionObserver === "undefined" ||
      typeof window.matchMedia !== "function"
    ) {
      return;
    }

    const sections = Array.from(
      root.querySelectorAll<HTMLElement>(".pulse-reveal"),
    );
    const revealed = new Set<Element>();
    const motion = window.matchMedia(
      "(prefers-reduced-motion: no-preference)",
    );
    let observer: IntersectionObserver | undefined;

    const clearClasses = () => {
      sections.forEach((section) => {
        section.classList.remove(
          "pulse-reveal-pending",
          "pulse-reveal-visible",
        );
      });
    };

    const syncMotionPreference = () => {
      observer?.disconnect();
      clearClasses();
      if (!motion.matches) return;

      observer = new IntersectionObserver(
        (entries, currentObserver) => {
          if (!motion.matches) return;
          entries.forEach((entry) => {
            if (!entry.isIntersecting || revealed.has(entry.target)) return;
            entry.target.classList.remove("pulse-reveal-pending");
            entry.target.classList.add("pulse-reveal-visible");
            revealed.add(entry.target);
            currentObserver.unobserve(entry.target);
          });
        },
        { rootMargin: "-60px", threshold: 0 },
      );

      sections.forEach((section) => {
        if (revealed.has(section)) return;
        section.classList.add("pulse-reveal-pending");
        observer?.observe(section);
      });
    };

    syncMotionPreference();
    motion.addEventListener("change", syncMotionPreference);
    return () => {
      motion.removeEventListener("change", syncMotionPreference);
      observer?.disconnect();
      clearClasses();
    };
  }, [rootRef]);
}
