import java.text.DecimalFormat;
import java.text.NumberFormat;
public class T04 {
    public static void main(String[] args) {
        NumberFormat formatter = new DecimalFormat("#0");
        System.out.println("Miles" + "\t" + "Kilometers");
        System.out.println("1" + "\t" + (1 * 1.609));
        System.out.println("2" + "\t" + (2 * 1.609));
        System.out.println("3" + "\t" + (3 * 1.609));
        System.out.println("4" + "\t" + (4 * 1.609));
        System.out.println("5" + "\t" + (5 * 1.609));
        System.out.println("6" + "\t" + (6 * 1.609));
        System.out.println("7" + "\t" + (7 * 1.609));
        System.out.println("8" + "\t" + (8 * 1.609));
        System.out.println("9" + "\t" + (9 * 1.609));
        System.out.println("10" + "\t" + (10 * 1.609) + formatter.format(0.0));
    }
}
