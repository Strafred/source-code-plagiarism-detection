import java.text.DecimalFormat;
import java.util.Scanner;
public class t02 {
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        System.out.print("Enter the radius and length of a cylinder : ");
        double radius = sc.nextDouble();
        double length = sc.nextDouble();
        double area = radius * radius * 3.14159;
        double volume = (radius * radius * 3.14159) * length;
        DecimalFormat areaFormat = new DecimalFormat("#.#####");
        DecimalFormat volumeFormat = new DecimalFormat("#.#");
        System.out.println("The area is " + areaFormat.format(area));
        System.out.println("The volume is " + volumeFormat.format(volume));
    }
}
