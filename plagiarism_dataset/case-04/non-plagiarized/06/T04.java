import java.util.Scanner;
public class T04 {
    public static void main(String[] args) {
        double kilo;
        double milToKilo = 1.609;
        System.out.println("Miles\tKilometers");
        for (int i = 1; i < 11; i++) {
            kilo = i * milToKilo;
            System.out.println(i + "\t" + kilo);
        }
    }
}
